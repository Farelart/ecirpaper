"""Full pinned E5-small preparation on the same Modal T4 resource class."""
import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'intfloat/e5-small-v2'
REVISION = 'ffb93f3bd4047442299a41ebb6fa998a38507c52'
app = modal.App('ecir-e5-full-t4')
volume = modal.Volume.from_name('ecir-splade-benchmark')
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.8.0', 'transformers==4.57.6', 'numpy==2.0.2')


@app.function(image=image, gpu='T4', cpu=(2,4), memory=(8192,8192), timeout=14400,
              max_containers=1, scaledown_window=2, retries=0, volumes={'/data':volume})
def encode():
    import hashlib, time
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path('/data'); out = root/'e5_t4'; out.mkdir(exist_ok=True)
    cfg = {'model':MODEL, 'revision':REVISION, 'gpu':'T4', 'precision':'float32',
           'batch_size':32, 'max_length':512, 'pooling':'masked mean + L2 normalization',
           'prefixes':{'products':'passage: ', 'queries':'query: '},
           'input_sha256':{kind:hashlib.sha256((root/f'{kind}.jsonl').read_bytes()).hexdigest()
                           for kind in ('products','queries')}}
    mp = out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != cfg:
        raise ValueError('Changed pinned encoding settings')
    mp.write_text(json.dumps(cfg,indent=2)+'\n'); volume.commit()
    start = time.perf_counter(); tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model = AutoModel.from_pretrained(MODEL,revision=REVISION).to('cuda').eval()
    torch.cuda.synchronize(); load_seconds = time.perf_counter()-start
    for kind,n in [('products',164900),('queries',8956)]:
        rows = [json.loads(line) for line in (root/f'{kind}.jsonl').open()]
        assert len(rows) == len({r['id'] for r in rows}) == n
        ids = [r['id'] for r in rows]; (out/f'{kind}_ids.json').write_text(json.dumps(ids)+'\n')
        for offset in range(0,n,2048):
            dest = out/f'{kind}_{offset:06d}.npy'; meta = dest.with_suffix('.json')
            if dest.exists() and meta.exists():
                assert json.loads(meta.read_text())['ids'] == ids[offset:offset+2048]
                continue
            vectors=[]; seconds=0
            for pos in range(offset,min(offset+2048,n),32):
                batch = rows[pos:min(pos+32,n)]
                torch.cuda.synchronize(); start=time.perf_counter()
                x=tok([cfg['prefixes'][kind]+r['text'] for r in batch],return_tensors='pt',
                      padding=True,truncation=True,max_length=512).to('cuda')
                with torch.inference_mode():
                    h=model(**x).last_hidden_state; mask=x['attention_mask'].unsqueeze(-1)
                    z=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),p=2,dim=1)
                    a=z.cpu().numpy()
                torch.cuda.synchronize(); seconds+=time.perf_counter()-start
                assert np.isfinite(a).all() and np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5)
                vectors.append(a)
            a=np.concatenate(vectors); np.save(dest,a)
            meta.write_text(json.dumps({'ids':ids[offset:offset+2048], 'processing_seconds':seconds,
                                        'shape':list(a.shape)})+'\n'); volume.commit()
            progress={'kind':kind,'completed':min(offset+2048,n),'total':n,
                      'last_block_seconds':seconds,'last_block_items_per_second':len(a)/seconds}
            (out/'progress.json').write_text(json.dumps(progress)+'\n'); volume.commit()
            print(json.dumps(progress),flush=True)
    summary={'complete':True,'configuration':cfg,'products':164900,'queries':8956,
             'model_load_seconds':load_seconds,
             'processing_seconds':{kind:sum(json.loads(p.read_text())['processing_seconds']
                for p in out.glob(f'{kind}_*.json') if p.name!=f'{kind}_ids.json')
                for kind in ('products','queries')},
             'timing_scope':'Tokenization, transfers, forward pass, pooling, normalization and CPU result transfer; CUDA synchronization; checkpoint I/O/model loading separate.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); volume.commit()
    return json.dumps(summary)


@app.local_entrypoint()
def main():
    out=ROOT/'outputs/modal_e5_full'; out.mkdir(parents=True,exist_ok=True)
    if (out/'call.json').exists():
        raise FileExistsError('Inspect saved call before any new invocation')
    call=encode.spawn()
    (out/'call.json').write_text(json.dumps({'call_id':call.object_id,
        'volume':'ecir-splade-benchmark','remote_directory':'e5_t4'},indent=2)+'\n')
    print('E5 T4 call:',call.object_id,flush=True)
