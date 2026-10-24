"""Rebuild indexes and benchmark all eight methods on one Modal T4 host."""
import json
from pathlib import Path
import modal
from modal_e5_full import image as base_image

ROOT=Path(__file__).resolve().parents[1]
app=modal.App('ecir-common-t4-retrieval')
data=modal.Volume.from_name('ecir-common-t4-benchmark',create_if_missing=True)
neural=modal.Volume.from_name('ecir-splade-benchmark')
image=(base_image.pip_install('scipy==1.16.2')
       .add_local_file(ROOT/'scripts/modal_e5_full.py','/root/modal_e5_full.py')
       .add_local_file(ROOT/'scripts/benchmark_final_methods.py','/root/benchmark_final_methods.py')
       .add_local_file(ROOT/'scripts/evaluate_neural_methods.py','/root/evaluate_neural_methods.py')
       .add_local_dir(ROOT/'src/ecirpaper','/src/ecirpaper'))


@app.function(image=image,gpu='T4',cpu=(2,4),memory=(8192,8192),timeout=14400,
              max_containers=1,scaledown_window=2,retries=0,
              volumes={'/data':data,'/neural':neural})
def benchmark():
    import os
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    import csv,gzip,hashlib,platform,socket,time
    import numpy as np
    import scipy.sparse as sp
    import torch
    from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM
    import benchmark_final_methods as engine
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    out=Path('/data/results');out.mkdir(exist_ok=True);engine.OUT=out
    manifest=json.loads(Path('/data/inputs_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((Path('/data')/name).read_bytes()).hexdigest()==digest
    e5=Path('/neural/e5_t4');splade=Path('/neural/encoded')
    assert json.loads((e5/'summary.json').read_text())['complete']
    assert json.loads((splade/'summary.json').read_text())['complete']
    pids=np.array(json.loads(Path('/data/products_ids.json').read_text()))
    qrows=[json.loads(line) for line in Path('/data/queries.jsonl').open()]
    reference=json.loads(Path('/data/query_tokens_reference.json').read_text())
    assert len(pids)==164900 and len(qrows)==8956
    for row in qrows:assert engine.tokenize(row['query'])==reference[row['query']]
    reports={};pmap={pid:i for i,pid in enumerate(pids)}
    def progress(stage,detail):
        (out/'progress.json').write_text(json.dumps({'stage':stage,'detail':detail})+'\n')
        data.commit();print(stage,detail,flush=True)
    def quality(name,score):
        if not (out/name/'quality.json').exists():engine.candidate_quality(name,score,qrows,pids)
    def finalize(name,result):
        result['hardware']={'gpu':torch.cuda.get_device_name(),'gpu_class':'T4',
            'cpu':platform.processor(),'container_hostname':socket.gethostname(),
            'cpu_threads':1,'cpu_request_cores':2,'cpu_limit_cores':4,
            'memory_limit_mib':8192,'host':platform.platform(),
            'torch':str(torch.__version__),'numpy':np.__version__,'scipy':__import__('scipy').__version__}
        result['scope']='Same Modal T4 host; exact full-corpus top10, single query, warm resident index, three shuffled repeats. Neural query encoding/dot search on GPU; score transfer and deterministic top10 selection on CPU included. BM25 uses CPU postings. No network/ANN.'
        (out/name/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        reports[name]=result;progress(name,'complete')
    for name in ('plain','structured','doc2query_t5','ling_doc2query'):
        dest=out/name;dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():reports[name]=json.loads((dest/'summary.json').read_text());continue
        progress(name,'building index');texts=[];token_seconds=0;mismatches=0
        with gzip.open('/data/lexical_inputs.jsonl.gz','rt') as stream:
            for i,line in enumerate(stream):
                row=json.loads(line);assert row['product_id']==pids[i]
                tokens=row['doc2query_t5'] if name=='doc2query_t5' else row['base']+(row[name] if name!='plain' else [])
                raw=row['raw_t5'] if name=='doc2query_t5' else row['raw_base']+(' '+row['raw_structured_expansion'] if name=='structured' else ' '+row['raw_ling_expansion'] if name=='ling_doc2query' else '')
                start=time.perf_counter();observed=engine.tokenize(raw);token_seconds+=time.perf_counter()-start
                mismatches+=int(observed!=tokens);texts.append(tokens)
        matrix,vocab,build=engine.bm25_index(texts,dest);del texts
        build['index_bytes']=(dest/'index.npz').stat().st_size+(dest/'vocabulary.json').stat().st_size
        build.update(document_tokenization_seconds=token_seconds,
            total_indexing_seconds=token_seconds+build['build_and_write_seconds'],
            frozen_token_stream_mismatching_documents=mismatches,
            note='Tokenization timed on source text separately; index built from frozen canonical tokens to preserve prior ranking semantics. Python runtime differences recorded, not silently applied.')
        score=lambda text:engine.bm25_scores(text,matrix,vocab)
        quality(name,lambda row:score(row['query']))
        finalize(name,engine.timings(name,score,pids,qrows,build));del matrix,vocab
    dense=None;dense_queries=None
    for name,modelid,revision,source in [
        ('e5','intfloat/e5-small-v2','ffb93f3bd4047442299a41ebb6fa998a38507c52',e5),
        ('splade','naver/splade-cocondenser-ensembledistil','49cf4c7b0db5b870a401ddf5e2669993ef3699c7',splade)]:
        dest=out/name;dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():reports[name]=json.loads((dest/'summary.json').read_text());continue
        progress(name,'assembling GPU index');start=time.perf_counter()
        ids=json.loads((source/'products_ids.json').read_text());mapping={p:i for i,p in enumerate(ids)}
        order=[mapping[p] for p in pids]
        if name=='e5':
            matrix=np.concatenate([np.load(p) for p in sorted(source.glob('products_*.npy'))])[order]
            np.save(dest/'index.npy',matrix);gpu=torch.from_numpy(matrix).to('cuda')
            qmatrix=np.concatenate([np.load(p) for p in sorted(source.glob('queries_*.npy'))])
        else:
            matrix=sp.vstack([sp.load_npz(p) for p in sorted(source.glob('products_*.npz'))],format='csr')[order];matrix.sort_indices()
            sp.save_npz(dest/'index.npz',matrix)
            gpu=torch.sparse_csr_tensor(torch.from_numpy(matrix.indptr.astype(np.int64)),
                torch.from_numpy(matrix.indices.astype(np.int64)),torch.from_numpy(matrix.data),
                size=matrix.shape,device='cuda')
            qmatrix=sp.vstack([sp.load_npz(p) for p in sorted(source.glob('queries_*.npz'))],format='csr')
        torch.cuda.synchronize();build={'build_and_write_seconds':time.perf_counter()-start,
            'index_bytes':sum(p.stat().st_size for p in dest.glob('index.*')),
            'note':'Representation assembly, serialization and upload to resident GPU included; product model encoding measured separately.'}
        tok=AutoTokenizer.from_pretrained(modelid,revision=revision)
        model=(AutoModel if name=='e5' else AutoModelForMaskedLM).from_pretrained(modelid,revision=revision).to('cuda').eval()
        def encode(query):
            x=tok(('query: '+query) if name=='e5' else query,return_tensors='pt',truncation=True,max_length=512).to('cuda')
            with torch.inference_mode():
                y=model(**x)
                if name=='e5':
                    mask=x['attention_mask'].unsqueeze(-1);z=torch.nn.functional.normalize((y.last_hidden_state*mask).sum(1)/mask.sum(1),p=2,dim=1)[0]
                else:z=torch.max(torch.log1p(torch.relu(y.logits))*x['attention_mask'].unsqueeze(-1),dim=1).values[0]
            torch.cuda.synchronize();return z
        def score(vector):
            with torch.inference_mode():
                value=(torch.mv(gpu,vector) if name=='e5' else torch.sparse.mm(gpu,vector[:,None]).ravel()).cpu().numpy()
            if not np.isfinite(value).all():raise ValueError('Nonfinite GPU scores')
            return value
        qids=json.loads((source/'queries_ids.json').read_text());qmap={q:i for i,q in enumerate(qids)}
        def stored_score(row):
            v=qmatrix[qmap[row['query_id']]]
            a=v if name=='e5' else v.toarray().ravel()
            return score(torch.from_numpy(np.asarray(a,dtype=np.float32)).to('cuda'))
        # Independent CPU dot-product check on real full-corpus query scores.
        for row in qrows[::max(1,len(qrows)//20)]:
            v=qmatrix[qmap[row['query_id']]]
            actual=stored_score(row)
            expected=np.einsum('ij,j->i',matrix,v,optimize=False) if name=='e5' else (matrix@v.T).toarray().ravel()
            assert np.allclose(actual,expected,rtol=1e-4,atol=1e-5)
        build['independent_score_validation']='CPU dot-product agrees for21 evenly spaced full-corpus queries; rtol1e-4 atol1e-5.'
        quality(name,stored_score);finalize(name,engine.timings(name,score,pids,qrows,build,encode))
        del model,gpu,matrix,qmatrix;torch.cuda.empty_cache()
    # Same CUDA dense query path for both hybrids; timed RRF uses1000 hits/component.
    ids=json.loads((e5/'products_ids.json').read_text());mapping={p:i for i,p in enumerate(ids)}
    dense=torch.from_numpy(np.concatenate([np.load(p) for p in sorted(e5.glob('products_*.npy'))])[[mapping[p] for p in pids]]).to('cuda')
    dense_queries=np.concatenate([np.load(p) for p in sorted(e5.glob('queries_*.npy'))])
    qi={q:i for i,q in enumerate(json.loads((e5/'queries_ids.json').read_text()))}
    tok=AutoTokenizer.from_pretrained('intfloat/e5-small-v2',revision='ffb93f3bd4047442299a41ebb6fa998a38507c52')
    model=AutoModel.from_pretrained('intfloat/e5-small-v2',revision='ffb93f3bd4047442299a41ebb6fa998a38507c52').to('cuda').eval()
    def hybrid_encode(query):
        x=tok('query: '+query,return_tensors='pt',truncation=True,max_length=512).to('cuda')
        with torch.inference_mode():
            h=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1)
            z=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),p=2,dim=1)[0]
        torch.cuda.synchronize();return query,z
    for lexical in ('plain','structured'):
        name='hybrid_'+lexical;dest=out/name;dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():reports[name]=json.loads((dest/'summary.json').read_text());continue
        matrix=sp.load_npz(out/lexical/'index.npz');vocab=json.loads((out/lexical/'vocabulary.json').read_text())
        def components(query,vector):
            lex=engine.bm25_scores(query,matrix,vocab);neural_scores=torch.mv(dense,vector).cpu().numpy()
            assert np.isfinite(neural_scores).all();return lex,neural_scores
        def fused(encoded):
            scores=np.zeros(len(pids))
            for component in components(*encoded):
                selected=engine.top(component,pids,1000);scores[selected]+=1/(60+np.arange(1,1001))
            return scores
        def quality_score(row):
            values=components(row['query'],torch.from_numpy(dense_queries[qi[row['query_id']]]).to('cuda'))
            candidates=[pmap[p] for p in row['judgments']];scores=np.zeros(len(pids))
            for v in values:
                ranking=sorted(candidates,key=lambda i:(-v[i],pids[i]))
                scores[ranking]+=1/(60+np.arange(1,len(ranking)+1))
            return scores
        quality(name,quality_score)
        build={'build_and_write_seconds':reports[lexical]['index']['build_and_write_seconds']+reports['e5']['index']['build_and_write_seconds'],
            'total_indexing_seconds':reports[lexical]['index']['total_indexing_seconds']+reports['e5']['index']['build_and_write_seconds'],
            'index_bytes':reports[lexical]['index']['index_bytes']+reports['e5']['index']['index_bytes'],
            'note':'Sum of component builds; shared indexes. Candidate quality uses full-candidate RRF, timed search uses top1000/component.'}
        finalize(name,engine.timings(name,fused,pids,qrows,build,hybrid_encode));del matrix,vocab
    summary={'complete':True,'products':164900,'queries':8956,'methods':reports,'input_manifest':manifest}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');progress('all','complete')
    return json.dumps(summary)


@app.local_entrypoint()
def main():
    out=ROOT/'outputs/modal_common_t4';out.mkdir(exist_ok=True)
    if (out/'call.json').exists():raise FileExistsError('Inspect saved call before launching another')
    call=benchmark.spawn();(out/'call.json').write_text(json.dumps({'call_id':call.object_id,
        'volume':'ecir-common-t4-benchmark','remote_directory':'results'},indent=2)+'\n')
    print('Common T4 benchmark call:',call.object_id,flush=True)
