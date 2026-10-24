"""Checkpointed full hosted T5 preparation timing; fixed single T4 worker."""
import hashlib
import json
import time
from pathlib import Path
import modal
from modal_t5_benchmark import image, _benchmark, MODEL, REVISION

ROOT=Path(__file__).resolve().parents[1]
image=image.add_local_file(ROOT/'scripts/modal_t5_benchmark.py','/root/modal_t5_benchmark.py')
app=modal.App('ecir-doc2query-t5-full')
volume=modal.Volume.from_name('ecir-t5-cloud-benchmark',create_if_missing=True)


@app.function(image=image,gpu='T4',cpu=(2,4),memory=(8192,8192),timeout=14400,
              max_containers=1,scaledown_window=2,retries=0,volumes={'/data':volume})
def run_full():
    started=time.perf_counter()
    data=Path('/data');source=data/'inputs.jsonl';out=data/'full_t4';out.mkdir(exist_ok=True)
    rows=[json.loads(line) for line in source.open()]
    if len(rows)!=164900 or len({r['product_id'] for r in rows})!=164900:
        raise ValueError('Full product coverage required')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    configuration={'model':MODEL,'revision':REVISION,'input_sha256':digest,
                   'products':164900,'gpu':'T4','batch_size':16,'queries_per_product':3,
                   'seed_policy':'42 + full sorted batch position','precision':'float32',
                   'input_max_tokens':512,'max_length':64,'top_k':10,'top_p':1.0,'temperature':1.0,
                   'cpu_request_cores':2,'cpu_limit_cores':4,'memory_limit_mib':8192}
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=configuration:
        raise ValueError('Checkpoint configuration changed')
    manifest.write_text(json.dumps(configuration,indent=2)+'\n');volume.commit()
    for offset in range(0,len(rows),2048):
        checkpoint=out/f'part_{offset:06d}.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            if [r['product_id'] for r in saved['products']]!=[r['product_id'] for r in rows[offset:offset+2048]]:
                raise ValueError('Checkpoint product mismatch')
            continue
        result=_benchmark(rows[offset:offset+2048],offset)
        checkpoint.write_text(json.dumps(result)+'\n')
        volume.commit()
        done=min(offset+2048,len(rows))
        (out/'progress.json').write_text(json.dumps({'completed':done,'total':len(rows),'last_checkpoint':checkpoint.name,'session_elapsed_seconds':time.perf_counter()-started})+'\n')
        volume.commit()
        print(f'Checkpointed {done}/{len(rows)} products',flush=True)
    parts=[json.loads((out/f'part_{offset:06d}.json').read_text()) for offset in range(0,len(rows),2048)]
    ids=[r['product_id'] for p in parts for r in p['products']]
    if ids!=[r['product_id'] for r in rows]:raise ValueError('Final coverage mismatch')
    seconds=sum(b['wall_seconds'] for p in parts for b in p['batches'])
    summary={'complete':True,'configuration':configuration,'products':len(ids),
             'batch_processing_seconds':seconds,'products_per_second':len(ids)/seconds,
             'session_wall_seconds':time.perf_counter()-started,
             'model_load_seconds':sum(p['model_load_seconds'] for p in parts),
             'warmup_seconds':sum(p['warmup_seconds'] for p in parts),
             'function_body_seconds_sum':sum(p['function_body_seconds'] for p in parts),
             'estimated_batch_compute_usd':seconds*(.000164+2*.0000131+8*.00000222),
             'actual_metered_compute_usd':None,
             'note':'Full CUDA generation for preparation measurements; keep existing local effectiveness rankings. Pricing estimate excludes startup and other billing phases; verify Modal app billing. Session elapsed time is not a resumed full-run total.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');volume.commit()
    return json.dumps(summary)


@app.local_entrypoint()
def main():
    out=ROOT/'outputs/modal_t5_full';out.mkdir(exist_ok=True)
    if (out/'call.json').exists():raise FileExistsError('Check existing cloud call before launching another')
    call=run_full.spawn()
    (out/'call.json').write_text(json.dumps({'call_id':call.object_id,'volume':'ecir-t5-cloud-benchmark','model':MODEL,'revision':REVISION,'products':164900},indent=2)+'\n')
    print('Full cloud call:',call.object_id,flush=True)
    # Detached mode keeps this one cloud invocation alive after local disconnect.
    try:
        result=call.get(timeout=30)
    except TimeoutError:
        print('Cloud run continues; retrieve result using the saved call ID or volume.',flush=True)
        return
    (out/'summary.json').write_text(result+'\n')
