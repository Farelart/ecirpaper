"""Pinned, checkpointed SPLADE encoding on one bounded Modal T4 worker."""
import json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
MODEL='naver/splade-cocondenser-ensembledistil'
REVISION='49cf4c7b0db5b870a401ddf5e2669993ef3699c7'
app=modal.App('ecir-splade-full')
volume=modal.Volume.from_name('ecir-splade-benchmark',create_if_missing=True)
image=modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.8.0','transformers==4.57.6','scipy==1.16.2')

@app.function(image=image,gpu='T4',cpu=(2,4),memory=(8192,8192),timeout=14400,
              max_containers=1,scaledown_window=2,retries=0,volumes={'/data':volume})
def encode():
    import hashlib,time
    import numpy as np
    import torch
    import scipy.sparse as sp
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    torch.set_num_threads(4)
    root=Path('/data');out=root/'encoded';out.mkdir(exist_ok=True)
    cfg={'model':MODEL,'revision':REVISION,'max_length':512,'batch_size':8,
         'precision':'float32','pooling':'max(log1p(relu(logits)) * attention_mask)',
         'pruning':'none; retain every strictly positive weight','gpu':'T4',
         'input_sha256':{kind:hashlib.sha256((root/f'{kind}.jsonl').read_bytes()).hexdigest() for kind in ('products','queries')}}
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=cfg:raise ValueError('Changed configuration')
    manifest.write_text(json.dumps(cfg,indent=2)+'\n');volume.commit()
    started=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModelForMaskedLM.from_pretrained(MODEL,revision=REVISION).to('cuda').eval()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-started
    def infer(rows):
        torch.cuda.synchronize();t=time.perf_counter()
        x=tok([r['text'] for r in rows],return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda')
        with torch.inference_mode():
            logits=model(**x).logits
            weights=torch.max(torch.log1p(torch.relu(logits))*x['attention_mask'].unsqueeze(-1),dim=1).values
            a=weights.cpu().numpy()
        torch.cuda.synchronize()
        return sp.csr_matrix(a),time.perf_counter()-t
    for kind,n in [('products',164900),('queries',8956)]:
        rows=[json.loads(l) for l in (root/f'{kind}.jsonl').open()];assert len(rows)==n
        ids=[r['id'] for r in rows];assert len(set(ids))==n
        (out/f'{kind}_ids.json').write_text(json.dumps(ids)+'\n')
        for offset in range(0,n,2048):
            path=out/f'{kind}_{offset:06d}.npz';meta=out/f'{kind}_{offset:06d}.json'
            if path.exists() and meta.exists():
                assert json.loads(meta.read_text())['ids']==ids[offset:offset+2048];continue
            mats=[];seconds=0
            for pos in range(offset,min(offset+2048,n),8):
                mat,elapsed=infer(rows[pos:min(pos+8,n)]);mats.append(mat);seconds+=elapsed
            matrix=sp.vstack(mats,format='csr');sp.save_npz(path,matrix)
            meta.write_text(json.dumps({'ids':ids[offset:offset+2048],'processing_seconds':seconds,'shape':list(matrix.shape),'nnz':matrix.nnz})+'\n')
            volume.commit();done=min(offset+2048,n)
            (out/'progress.json').write_text(json.dumps({'kind':kind,'completed':done,'total':n,'last_block_seconds':seconds,'last_block_products_per_second':matrix.shape[0]/seconds})+'\n');volume.commit()
            print(f'Checkpointed {kind} {done}/{n}; {matrix.shape[0]/seconds:.2f}/s',flush=True)
    summary={'complete':True,'configuration':cfg,'model_load_seconds_latest_session':load_seconds,
             'products':164900,'queries':8956,'processing_seconds':{kind:sum(json.loads(p.read_text())['processing_seconds'] for p in out.glob(f'{kind}_*.json') if p.name!=f'{kind}_ids.json') for kind in ('products','queries')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');volume.commit();return json.dumps(summary)

@app.local_entrypoint()
def main():
    out=ROOT/'outputs/splade';out.mkdir(parents=True,exist_ok=True)
    if (out/'call.json').exists():raise FileExistsError('Check existing Modal call before launching')
    c=encode.spawn();(out/'call.json').write_text(json.dumps({'call_id':c.object_id,'volume':'ecir-splade-benchmark'},indent=2)+'\n')
    print('SPLADE call:',c.object_id,flush=True)
    try:r=c.get(timeout=30)
    except TimeoutError:print('Cloud encoding continues',flush=True);return
    (out/'summary.json').write_text(r+'\n')
