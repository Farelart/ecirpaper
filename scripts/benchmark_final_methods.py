"""Full-corpus single-query CPU retrieval timing and new candidate evaluations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['VECLIB_MAXIMUM_THREADS']='1'
import csv,json,sys,time,hashlib,shutil
from array import array
from collections import Counter
from pathlib import Path
import numpy as np
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ecirpaper.retrieval import tokenize
from ecirpaper.metrics import evaluate_ranking
from evaluate_neural_methods import summarize,interval,load_rankings
OUT=ROOT/'outputs/final_benchmark'

def top(scores,pids,k=10):
    threshold=np.partition(scores,-k)[-k]
    selected=np.flatnonzero(scores>=threshold)
    order=np.lexsort((pids[selected],-scores[selected]))
    return selected[order[:k]]

def bm25_index(texts,out):
    started=time.perf_counter();vocab={};indptr=array('q',[0]);indices=array('i');values=array('d');lengths=[]
    for i,text in enumerate(texts):
        counts=Counter(text if isinstance(text,list) else tokenize(text));lengths.append(sum(counts.values()))
        for term,f in counts.items():
            if term not in vocab:vocab[term]=len(vocab)
            indices.append(vocab[term]);values.append(f)
        indptr.append(len(indices))
        if (i+1)%25000==0:print('Indexed',i+1,'products',flush=True)
    m=sp.csr_matrix((np.frombuffer(values,dtype=np.float64),np.frombuffer(indices,dtype=np.int32),np.frombuffer(indptr,dtype=np.int64)),shape=(len(texts),len(vocab)))
    df=np.bincount(m.indices,minlength=m.shape[1]);idf=np.log1p((len(texts)-df+.5)/(df+.5));lengths=np.asarray(lengths)
    norm=1.2*(.25+.75*lengths/lengths.mean())
    for start in range(0,len(texts),4096):
        a,b=m.indptr[start],m.indptr[min(start+4096,len(texts))]
        denominator=m.data[a:b]+np.repeat(norm[start:start+4096],np.diff(m.indptr[start:min(start+4096,len(texts))+1]))
        m.data[a:b]=idf[m.indices[a:b]]*m.data[a:b]*2.2/denominator
    m=m.tocsc();m.sort_indices()
    sp.save_npz(out/'index.npz',m);(out/'vocabulary.json').write_text(json.dumps(vocab))
    return m,vocab,{'build_and_write_seconds':time.perf_counter()-started,'index_bytes':sum(p.stat().st_size for p in out.iterdir() if p.is_file()),'average_document_length':float(lengths.mean())}

def bm25_scores(query,m,vocab):
    cols=[vocab[t] for t in sorted(set(tokenize(query))) if t in vocab]
    return posting_scores(m,cols,np.ones(len(cols)))

def posting_scores(m,cols,weights):
    scores=np.zeros(m.shape[0],dtype=m.dtype)
    for col,weight in zip(cols,weights):
        start,end=m.indptr[col:col+2]
        scores[m.indices[start:end]]+=m.data[start:end]*weight
    return scores

def dense_scores(matrix,vector):
    # This macOS NumPy backend emits spurious floating-point flag warnings;
    # compare against independent einsum before timing and reject invalid output.
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        scores=np.asarray(matrix@vector).ravel()
    if not np.isfinite(scores).all() or np.max(np.abs(scores))>1.0001:
        raise ValueError('Invalid dense cosine scores')
    return scores

def timings(method,score,pids,qrows,build,query_encoder=None):
    out=OUT/method;out.mkdir(exist_ok=True);rng=np.random.default_rng(42)
    for row in qrows[:20]:
        value=query_encoder(row['query']) if query_encoder is not None else row['query']
        top(score(value),pids)
    elapsed=[];search=[];encoding=[]
    with (out/'latencies.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['repeat','query_id','total_seconds','encoding_seconds','search_and_top10_seconds'])
        for repeat in range(3):
            for pos in rng.permutation(len(qrows)):
                row=qrows[pos];start=time.perf_counter();enc=0
                if query_encoder is not None:
                    vector=query_encoder(row['query']);enc=time.perf_counter()-start;scores=score(vector)
                else:scores=score(row['query'])
                top(scores,pids);duration=time.perf_counter()-start
                elapsed.append(duration);encoding.append(enc);search.append(duration-enc)
                writer.writerow([repeat,row['query_id'],duration,enc,duration-enc])
            print(method,'retrieval repeat',repeat+1,'complete',flush=True)
    report={'method':method,'products':len(pids),'queries':len(qrows),'repeats':3,'warmup_queries':20,
        'median_ms':float(np.median(elapsed)*1000),'p95_ms':float(np.quantile(elapsed,.95)*1000),
        'mean_query_encoding_ms':float(np.mean(encoding)*1000),'mean_search_and_top10_ms':float(np.mean(search)*1000),
        'index':build,'hardware':'Local Mac; single CPU thread; all methods sequential; warm resident indexes',
        'scope':'Full 164900-product exact search, top10 selection, single query at a time; neural encoding included. No ANN or serving/network layer.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report

def candidate_quality(name,score,qrows,pids):
    positions={pid:i for i,pid in enumerate(pids)};rows=[];out=OUT/name
    with (out/'rankings.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['query_id','product_id','rank','score'])
        for row in qrows:
            scores=score(row);ranking=sorted(row['judgments'],key=lambda pid:(-scores[positions[pid]],pid))
            rows.append({'query_id':row['query_id'],**evaluate_ranking(ranking,row['judgments'])})
            for rank,pid in enumerate(ranking,1):w.writerow([row['query_id'],pid,rank,format(float(scores[positions[pid]]),'.17g')])
    results=summarize(rows);(out/'quality.json').write_text(json.dumps({'results':{name:results}},indent=2)+'\n')
    return results

def main():
    OUT.mkdir(exist_ok=True)
    qrows=[json.loads(l) for l in (ROOT/'outputs/neural_inputs/queries.jsonl').open()]
    originals={r['product_id']:r['text'] for r in (json.loads(l) for l in (ROOT/'outputs/final_benchmark_inputs/products_bm25.jsonl').open())}
    t5_originals={r['product_id']:r['text'] for r in (json.loads(l) for l in (ROOT/'outputs/neural_inputs/products.jsonl').open())}
    canonical={r['product_id']:r for r in (json.loads(l) for l in (ROOT/'outputs/final_benchmark_inputs/canonical_tokens.jsonl').open())}
    tokenization_times=json.loads((ROOT/'outputs/final_benchmark_inputs/tokenization_timing.json').read_text())
    query_reference=json.loads((ROOT/'outputs/final_benchmark_inputs/query_tokens_reference.json').read_text())
    for row in qrows:
        if tokenize(row['query'])!=query_reference[row['query']]:raise ValueError('Query tokenizer differs across runtimes')
    pids=np.array(sorted(originals));assert len(pids)==164900 and len(qrows)==8956
    variants={'plain':{},'structured':{},'doc2query_t5':{},'ling_doc2query':{}}
    for l in (ROOT/'outputs/budget_enrichment/test/products_enriched.jsonl').open():
        r=json.loads(l);variants['structured'][r['product_id']]=' '.join(s for f in r['enrichment'].values() for s in f)
    for name,path in [('doc2query_t5','outputs/doc2query_t5/full_b16/products.jsonl'),('ling_doc2query','outputs/budget_enrichment/ling_doc2query/products_queries.jsonl')]:
        for l in (ROOT/path).open():
            r=json.loads(l);variants[name][r['product_id']]=' '.join(r['queries'])
        assert set(variants[name])==set(originals)
    reports={}
    for name,expansion in variants.items():
        out=OUT/name;out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():reports[name]=json.loads((out/'summary.json').read_text());continue
        texts=[t5_originals[pid]+' '+expansion.get(pid,'') for pid in pids] if name=='doc2query_t5' else [canonical[pid]['base']+(canonical[pid][name] if name!='plain' else []) for pid in pids]
        m,vocab,build=bm25_index(texts,out)
        build['document_tokenization_seconds_separately_measured']=0 if name=='doc2query_t5' else tokenization_times['base_seconds']+tokenization_times['additional_expansion_seconds'].get(name,0)
        build['total_indexing_seconds']=build['build_and_write_seconds']+build['document_tokenization_seconds_separately_measured']
        build['text_preparation_note']='Canonical document tokens from frozen Python 3.12 tokenizer; all query tokenizations validated identical under timing runtime. Document tokenization performed separately and excluded from index assembly time.' if name!='doc2query_t5' else 'Previously cleaned neural export under frozen Python 3.9 T5 scorer. No previous rankings changed.'
        score=lambda q:bm25_scores(q,m,vocab)
        # Verify actual index scores against saved reference scores before timing.
        if name!='ling_doc2query':
            path=ROOT/({'plain':'outputs/enrichment_evaluation/plain_rankings.csv','structured':'outputs/enrichment_evaluation/all_rankings.csv','doc2query_t5':'outputs/doc2query_evaluation/doc2query_t5_rankings.csv'}[name])
            references={};selected={r['query_id']:r for r in qrows[:100]};mapping={pid:i for i,pid in enumerate(pids)}
            for r in csv.DictReader(path.open()):
                qid=int(r['query_id'])
                if qid in selected:references.setdefault(qid,[]).append(r)
            for qid,records in references.items():
                scores=score(selected[qid]['query'])
                for r in records:
                    ref=float(r.get('bm25_score',r.get('score')));assert np.isclose(scores[mapping[r['product_id']]],ref,rtol=1e-12,atol=1e-12),(name,qid,r)
            build['reference_score_validation_queries']=100
        else:candidate_quality(name,lambda row:score(row['query']),qrows,pids)
        reports[name]=timings(name,score,pids,qrows,build)
        del m,vocab
    import torch
    from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM
    torch.set_num_threads(1)
    for name,modelid,revision in [('e5','intfloat/e5-small-v2','ffb93f3bd4047442299a41ebb6fa998a38507c52'),('splade','naver/splade-cocondenser-ensembledistil','49cf4c7b0db5b870a401ddf5e2669993ef3699c7')]:
        out=OUT/name;out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():reports[name]=json.loads((out/'summary.json').read_text());continue
        start=time.perf_counter()
        if name=='e5':
            source=ROOT/'outputs/e5_small';ids=json.loads((source/'products_ids.json').read_text());v=np.load(source/'products_vectors.npy');mapping={pid:i for i,pid in enumerate(ids)};matrix=v[[mapping[pid] for pid in pids]].copy();np.save(out/'index.npy',matrix)
        else:
            source=ROOT/'outputs/splade/checkpoints/encoded';ids=json.loads((source/'products_ids.json').read_text());matrix=sp.vstack([sp.load_npz(p) for p in sorted(source.glob('products_*.npz'))],format='csr');mapping={pid:i for i,pid in enumerate(ids)};matrix=matrix[[mapping[pid] for pid in pids]].tocsc();matrix.sort_indices();sp.save_npz(out/'index.npz',matrix)
        build={'build_and_write_seconds':time.perf_counter()-start,'index_bytes':sum(p.stat().st_size for p in out.iterdir() if p.is_file()),'note':'Representation assembly/serialization only; document model encoding reported separately.'}
        if name=='e5':
            if not np.isfinite(matrix).all():raise ValueError('Invalid dense index')
            reference_vectors=np.load(source/'queries_vectors.npy',mmap_mode='r')
            for pos in range(0,len(reference_vectors),max(1,len(reference_vectors)//20)):
                actual=dense_scores(matrix,reference_vectors[pos])
                expected=np.einsum('ij,j->i',matrix,reference_vectors[pos],optimize=False)
                if not np.allclose(actual,expected,rtol=1e-5,atol=1e-6):raise ValueError('Dense backend score mismatch')
            build['dense_backend_validation']='Finite cosine bounds checked on every query; independent einsum agrees for 20+ reference queries across full corpus within float32 tolerance.'
        tok=AutoTokenizer.from_pretrained(modelid,revision=revision);model=(AutoModel if name=='e5' else AutoModelForMaskedLM).from_pretrained(modelid,revision=revision).eval()
        def encode(query):
            x=tok(('query: '+query) if name=='e5' else query,return_tensors='pt',truncation=True,max_length=512)
            with torch.inference_mode():
                y=model(**x)
                if name=='e5':
                    mask=x['attention_mask'].unsqueeze(-1);z=(y.last_hidden_state*mask).sum(1)/mask.sum(1);z=torch.nn.functional.normalize(z,p=2,dim=1);return z.numpy()[0]
                z=torch.max(torch.log1p(torch.relu(y.logits))*x['attention_mask'].unsqueeze(-1),dim=1).values.numpy();return sp.csr_matrix(z).T
        score=lambda vector:dense_scores(matrix,vector) if name=='e5' else posting_scores(matrix,vector.indices if sp.isspmatrix_csc(vector) else vector.tocsc().indices,vector.data if sp.isspmatrix_csc(vector) else vector.tocsc().data)
        if name=='splade':
            qids=json.loads((source/'queries_ids.json').read_text());qm=sp.vstack([sp.load_npz(p) for p in sorted(source.glob('queries_*.npz'))],format='csr');qmap={qid:i for i,qid in enumerate(qids)}
            candidate_quality(name,lambda row:score(qm[qmap[row['query_id']]].T),qrows,pids)
        reports[name]=timings(name,score,pids,qrows,build,encode)
        del model,matrix
    # Fixed RRF over 1000 hits from each full-corpus component. This serving
    # benchmark is separate from full-candidate RRF effectiveness evaluation.
    source=ROOT/'outputs/e5_small';ids=json.loads((source/'products_ids.json').read_text());mapping={pid:i for i,pid in enumerate(ids)}
    dense=np.load(source/'products_vectors.npy')[[mapping[pid] for pid in pids]]
    tok=AutoTokenizer.from_pretrained('intfloat/e5-small-v2',revision='ffb93f3bd4047442299a41ebb6fa998a38507c52')
    model=AutoModel.from_pretrained('intfloat/e5-small-v2',revision='ffb93f3bd4047442299a41ebb6fa998a38507c52').eval()
    def hybrid_encode(query):
        x=tok('query: '+query,return_tensors='pt',truncation=True,max_length=512)
        with torch.inference_mode():
            y=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1)
            z=torch.nn.functional.normalize((y*mask).sum(1)/mask.sum(1),p=2,dim=1).numpy()[0]
        return query,z
    for lexical in ('plain','structured'):
        name='hybrid_'+lexical;out=OUT/name;out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():reports[name]=json.loads((out/'summary.json').read_text());continue
        m=sp.load_npz(OUT/lexical/'index.npz');vocab=json.loads((OUT/lexical/'vocabulary.json').read_text())
        def fused(encoded):
            query,vector=encoded;lex=bm25_scores(query,m,vocab);neural=dense_scores(dense,vector)
            result=np.zeros(len(pids))
            for scores in (lex,neural):
                selected=top(scores,pids,1000);result[selected]+=1/(60+np.arange(1,1001))
            return result
        # timings warmup passes raw text; use an adapter with the same full path.
        score=lambda value:fused(hybrid_encode(value) if isinstance(value,str) else value)
        build={'build_and_write_seconds':reports[lexical]['index']['build_and_write_seconds']+reports['e5']['index']['build_and_write_seconds'],
               'index_bytes':reports[lexical]['index']['index_bytes']+reports['e5']['index']['index_bytes'],
               'note':'Sum of separately measured component builds; shared indexes; RRF k60 with 1000 hits per component, fixed without test tuning.'}
        reports[name]=timings(name,score,pids,qrows,build,hybrid_encode)
    (ROOT/'reports/retrieval_benchmark.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
