"""Shared pinned reranker: full official-candidate quality and sampled full-corpus latency."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.8.0','transformers==4.57.6','numpy==2.0.2','scipy==1.16.2')
       .add_local_file(ROOT/'scripts/benchmark_final_methods.py','/root/benchmark_final_methods.py')
       .add_local_file(ROOT/'scripts/evaluate_neural_methods.py','/root/evaluate_neural_methods.py')
       .add_local_dir(ROOT/'src/ecirpaper','/src/ecirpaper'))
MODEL='cross-encoder/ms-marco-MiniLM-L-6-v2'
REVISION='233902d25c440f23af6f7d6e94d2946bac0bee0a'
METHODS=['plain','structured','ling_doc2query','e5','splade']
CONFIG={'model':MODEL,'revision':REVISION,'precision':'float16','max_length':256,
        'batch_size':32,'candidate_sizes':[10,20,50,100],'methods':METHODS,
        'quality_queries':8956,'quality_scope':'official judged-candidate ranking; rerank first min(N,candidates), append untouched tail',
        'timing_queries':100,'timing_repeats':3,'timing_seed':20261006,
        'timing_scope':'exact full-corpus retrieval + fresh neural query encoding + original-metadata reranking; warm single query; excludes network/load',
        'reranker_input':'original full cleaned metadata; tokenized query/document pair truncated to256 tokens; no enrichment supplied',
        'training':'no ESCI training or tuning','gpu':'T4','cpu_threads':1}
app=modal.App('ecir-shared-reranker-t4')
common=modal.Volume.from_name('ecir-common-t4-benchmark')
neural=modal.Volume.from_name('ecir-splade-benchmark')

@app.function(image=image,gpu='T4',cpu=(2,4),memory=(8192,8192),timeout=3600,
              retries=0,max_containers=1,scaledown_window=2,
              volumes={'/data':common,'/neural':neural})
def benchmark():
    import os
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    import csv,hashlib,time,math
    import numpy as np
    import scipy.sparse as sp
    import torch
    from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM,AutoModelForSequenceClassification
    import benchmark_final_methods as engine
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();out=Path('/data/rerank_results');out.mkdir(exist_ok=True)
    manifest=dict(CONFIG,input_hashes={name:hashlib.sha256(Path('/neural',name+'.jsonl').read_bytes()).hexdigest() for name in ('products','queries')})
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));common.commit()
    def progress(stage,**info):
        record={'stage':stage,'elapsed_seconds':time.perf_counter()-start,**info}
        (out/'progress.json').write_text(json.dumps(record,indent=2));common.commit();print(json.dumps(record),flush=True)
    def deadline():
        if time.perf_counter()-start>3450:raise TimeoutError('Hour target reached; checkpoints retained, no automatic retry')
    pids=np.array(json.loads(Path('/data/products_ids.json').read_text()))
    docs={str(r['id']):r['text'] for r in map(json.loads,Path('/neural/products.jsonl').open())}
    rows=list(map(json.loads,Path('/data/queries.jsonl').open()));qmap={r['query_id']:r for r in rows}
    assert len(docs)==len(pids)==164900 and len(rows)==8956
    baseline={}
    for method in METHODS:
        rank={}
        with Path('/data/results',method,'rankings.csv').open() as f:
            for r in csv.DictReader(f):rank.setdefault(int(r['query_id']),[]).append((int(r['rank']),r['product_id']))
        baseline[method]={q:[p for _,p in sorted(values)] for q,values in rank.items()}
        assert set(rank)==set(qmap)
        for row in rows:assert set(baseline[method][row['query_id']])==set(row['judgments'])
    pairs=[(r['query_id'],p) for r in rows for p in sorted(r['judgments'])]
    assert len(pairs)==181701 and len(set(pairs))==len(pairs)
    progress('loading_reranker',pairs=len(pairs))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float16).to('cuda').eval()
    def rerank(query,products):
        values=[]
        for offset in range(0,len(products),32):
            batch=products[offset:offset+32]
            x=tok([query]*len(batch),[docs[p] for p in batch],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
            with torch.inference_mode():scores=model(**x).logits.flatten().float().cpu().numpy()
            assert np.isfinite(scores).all();values.extend(scores.tolist())
        torch.cuda.synchronize();return values
    # Pilot is also the first saved quality block; no duplicated model scoring.
    scores={};process_seconds=0
    for offset in range(0,len(pairs),2048):
        deadline();path=out/f'pair_scores_{offset:06d}.json';batch=pairs[offset:offset+2048]
        if path.exists():saved=json.loads(path.read_text());assert saved['pairs']==[list(p) for p in batch]
        else:
            t=time.perf_counter();vals=[]
            for j in range(0,len(batch),32):
                part=batch[j:j+32]
                x=tok([qmap[q]['query'] for q,p in part],[docs[p] for q,p in part],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
                with torch.inference_mode():v=model(**x).logits.flatten().float().cpu().numpy()
                assert np.isfinite(v).all();vals.extend(v.tolist())
            torch.cuda.synchronize();saved={'pairs':batch,'scores':vals,'processing_seconds':time.perf_counter()-t}
            path.write_text(json.dumps(saved));common.commit()
        process_seconds+=saved['processing_seconds'];scores.update({tuple(p):s for p,s in zip(saved['pairs'],saved['scores'])})
        count=offset+len(batch);speed=count/process_seconds
        progress('quality_scoring',completed_pairs=count,total_pairs=len(pairs),pairs_per_second=speed,estimated_quality_remaining_seconds=(len(pairs)-count)/speed)
    with (out/'pair_scores.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['query_id','product_id','score'])
        for pair in pairs:w.writerow([*pair,format(scores[pair],'.17g')])
    gains={'E':1,'S':.1,'C':.01,'I':0}
    def ndcg(order,judgments):
        ideal=sorted((gains[v] for v in judgments.values()),reverse=True)[:10]
        denominator=sum(g/math.log2(i+2) for i,g in enumerate(ideal))
        return sum(gains[judgments[p]]/math.log2(i+2) for i,p in enumerate(order[:10]))/denominator if denominator else 0
    quality=[]
    for method in METHODS:
        for n in CONFIG['candidate_sizes']:
            for r in rows:
                q=r['query_id'];original=baseline[method][q];short=original[:n]
                ranking=sorted(short,key=lambda p:(-scores[q,p],p))+original[n:]
                relevant=sum(gains[v]>0 for v in r['judgments'].values())
                quality.append([method,n,q,ndcg(ranking,r['judgments']),ndcg(original,r['judgments']),sum(gains[r['judgments'][p]]>0 for p in short)/relevant if relevant else 0,len(short)])
    with (out/'quality.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','candidate_size','query_id','ndcg_at_10','baseline_ndcg_at_10','judged_relevant_candidate_recall','actual_candidates']);w.writerows(quality)
    common.commit();progress('quality_complete',queries=8956,methods=5,settings=20)
    rng=np.random.default_rng(CONFIG['timing_seed']);sample=[rows[i] for i in rng.choice(len(rows),100,replace=False)]
    (out/'timing_query_ids.json').write_text(json.dumps([r['query_id'] for r in sample]));common.commit()
    for method in METHODS:
        deadline();dest=out/f'{method}_latencies.csv'
        if dest.exists() and sum(1 for _ in dest.open())==1201:continue
        progress('timing_loading',method=method)
        query_model=None;gpu=None
        if method in ('plain','structured','ling_doc2query'):
            matrix=sp.load_npz(Path('/data/results',method,'index.npz'));vocab=json.loads(Path('/data/results',method,'vocabulary.json').read_text())
            def retrieve(query):return engine.top(engine.bm25_scores(query,matrix,vocab),pids,100)
        else:
            if method=='e5':
                matrix=np.load('/data/results/e5/index.npy');gpu=torch.from_numpy(matrix).to('cuda')
                mid='intfloat/e5-small-v2';rev='ffb93f3bd4047442299a41ebb6fa998a38507c52';cls=AutoModel
            else:
                matrix=sp.load_npz('/data/results/splade/index.npz').tocsr()
                gpu=torch.sparse_csr_tensor(torch.from_numpy(matrix.indptr.astype(np.int64)),torch.from_numpy(matrix.indices.astype(np.int64)),torch.from_numpy(matrix.data),size=matrix.shape,device='cuda')
                mid='naver/splade-cocondenser-ensembledistil';rev='49cf4c7b0db5b870a401ddf5e2669993ef3699c7';cls=AutoModelForMaskedLM
            query_tok=AutoTokenizer.from_pretrained(mid,revision=rev);query_model=cls.from_pretrained(mid,revision=rev).to('cuda').eval()
            def retrieve(query):
                x=query_tok(('query: '+query) if method=='e5' else query,truncation=True,max_length=512,return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    y=query_model(**x)
                    if method=='e5':
                        mask=x['attention_mask'].unsqueeze(-1);v=torch.nn.functional.normalize((y.last_hidden_state*mask).sum(1)/mask.sum(1),p=2,dim=1)[0];s=torch.mv(gpu,v)
                    else:v=torch.max(torch.log1p(torch.relu(y.logits))*x['attention_mask'].unsqueeze(-1),dim=1).values[0];s=torch.sparse.mm(gpu,v[:,None]).flatten()
                    s=s.cpu().numpy()
                assert np.isfinite(s).all();return engine.top(s,pids,100)
        for r in sample[:5]:rerank(r['query'],list(pids[retrieve(r['query'])[:10]]))
        with dest.open('w') as f:
            w=csv.writer(f);w.writerow(['method','candidate_size','repeat','query_id','retrieval_seconds','reranking_seconds','total_seconds'])
            for repeat in range(3):
                for i in rng.permutation(len(sample)):
                    r=sample[i];deadline()
                    for n in rng.permutation(CONFIG['candidate_sizes']):
                        torch.cuda.synchronize();t=time.perf_counter();selected=retrieve(r['query'])[:n];products=list(pids[selected]);t1=time.perf_counter()
                        vals=rerank(r['query'],products);sorted(zip(products,vals),key=lambda x:(-x[1],x[0]))[:10]
                        torch.cuda.synchronize();t2=time.perf_counter();w.writerow([method,int(n),repeat,r['query_id'],t1-t,t2-t1,t2-t])
                f.flush();common.commit();progress('timing',method=method,completed_repeats=repeat+1,total_repeats=3,queries_per_repeat=100)
        del matrix,query_model,gpu;torch.cuda.empty_cache()
    summary={'complete':True,'manifest':manifest,'quality_pairs':len(pairs),'quality_queries':len(rows),'quality_processing_seconds':process_seconds,'elapsed_seconds':time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));common.commit();progress('complete',**summary);return json.dumps(summary)

@app.local_entrypoint()
def main():
    out=ROOT/'outputs/modal_rerank';out.mkdir(exist_ok=True)
    if (out/'call.json').exists():raise FileExistsError('Saved call exists; inspect before recovery')
    call=benchmark.spawn();(out/'call.json').write_text(json.dumps({'call_id':call.object_id,'volume':'ecir-common-t4-benchmark','remote_directory':'rerank_results','config':CONFIG},indent=2))
    print('Reranker call:',call.object_id,flush=True)
