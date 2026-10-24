"""Bounded hosted T5 preparation pilot; no permanent deployment or test tuning."""
import hashlib
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'castorini/doc2query-t5-base-msmarco'
REVISION = '53926b5364548f0c81b39a31723184d3664aaf06'
image = modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.8.0', 'transformers==4.57.6', 'sentencepiece==0.2.1')
app = modal.App('ecir-doc2query-t5-benchmark')
_MODEL_CACHE = None


def _benchmark(rows, offset=0):
    global _MODEL_CACHE
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    if not 16 <= len(rows) <= 2048:
        raise ValueError('Pilot size outside bounded scope')
    torch.set_num_threads(4)
    start = time.perf_counter()
    if _MODEL_CACHE is None:
        print('Loading pinned tokenizer', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        print('Loading pinned model', flush=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, revision=REVISION).to('cuda').eval()
        _MODEL_CACHE = tokenizer, model
    else:
        tokenizer, model = _MODEL_CACHE
    print('Model loaded on GPU', flush=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start

    def infer(batch, seed):
        torch.cuda.synchronize()
        start = time.perf_counter()
        texts = ['\n'.join(str(v) for v in r['input'].values() if v) for r in batch]
        encoded = tokenizer(texts, return_tensors='pt', padding=True,
                            truncation=True, max_length=512, pad_to_multiple_of=128).to('cuda')
        torch.manual_seed(seed)
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=True, top_k=10,
                                    top_p=1.0, temperature=1.0, max_length=64,
                                    num_return_sequences=3)
        torch.cuda.synchronize()
        output = output.cpu()
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return decoded, time.perf_counter()-start, int(encoded['attention_mask'].sum().item())

    _, warmup_seconds, _ = infer(rows[:16], 42)
    batches, products = [], []
    for position in range(0, len(rows), 16):
        batch = rows[position:position+16]
        decoded, seconds, tokens = infer(batch, 42+offset+position)
        batches.append({'position': offset+position, 'products': len(batch),
                        'wall_seconds': seconds, 'input_tokens': tokens})
        products.extend({'product_id': r['product_id'], 'queries': decoded[i*3:(i+1)*3]}
                        for i, r in enumerate(batch))
        print(f'Generated {position+len(batch)}/{len(rows)} products; batch {seconds:.3f}s', flush=True)
    return {'model': MODEL, 'revision': REVISION, 'products': products,
            'batches': batches, 'model_load_seconds': load_seconds,
            'warmup_seconds': warmup_seconds, 'gpu': torch.cuda.get_device_name(),
            'torch': str(torch.__version__), 'transformers': str(transformers.__version__),
            'batch_size': 16, 'queries_per_product': 3, 'precision': 'float32',
            'function_body_seconds': time.perf_counter()-start,
            'input_max_tokens': 512, 'max_length': 64, 'top_k': 10,
            'top_p': 1.0, 'temperature': 1.0,
            'note': 'Same generation settings as local run; CUDA sampling need not reproduce MPS text exactly. Pilot is for throughput/cost, not replacement effectiveness rankings.'}


@app.function(image=image, gpu='T4', cpu=2, memory=8192, timeout=300,
              max_containers=1, scaledown_window=30, retries=0)
def benchmark(rows, offset=0):
    import traceback
    try:
        return json.dumps(_benchmark(rows, offset))
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        return json.dumps({'error': error})


@app.local_entrypoint()
def main(limit: int = 512, output: str = 'outputs/modal_t5_pilot/pilot_512'):
    if not 16 <= limit <= 2048:
        raise ValueError('Use a bounded pilot of 16 to 2048 products')
    source = ROOT/'outputs/budget_enrichment/test/products_enriched.jsonl'
    rows = [json.loads(line) for line in source.open() if line.strip()]
    assert len(rows) == 164900 and len({r['product_id'] for r in rows}) == 164900
    rows.sort(key=lambda r: hashlib.sha256(('t5-throughput-42:'+r['product_id']).encode()).digest())
    selected = [{'product_id': r['product_id'], 'input': r['input']} for r in rows[:limit]]
    out = ROOT/output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError('Preserve prior pilot; no duplicate billed run')
    (out/'selection.json').write_text(json.dumps(selected))
    start = time.perf_counter()
    parts=[]
    for offset in range(0, limit, 2048):
        print(f'Submitting request at product {offset}', flush=True)
        call=benchmark.spawn(selected[offset:offset+2048], offset)
        (out/'call.json').write_text(json.dumps({'call_id':call.object_id,'products':limit,'model':MODEL,'revision':REVISION},indent=2)+'\n')
        print(f'Function call: {call.object_id}', flush=True)
        try:
            part=json.loads(call.get(timeout=240))
        except TimeoutError:
            call.cancel(terminate_containers=True)
            (out/f'error_{time.time_ns()}.json').write_text(json.dumps({'error':'No result within 240 seconds; call cancelled','call_id':call.object_id,'offset':offset},indent=2)+'\n')
            raise
        (out/f'part_{offset:06d}.json').write_text(json.dumps(part, indent=2)+'\n')
        if 'error' in part:
            raise RuntimeError(part['error'])
        parts.append(part)
        print(f'Cloud pilot checkpoint: {offset+len(part["products"])}/{limit}', flush=True)
    result=dict(parts[0])
    result['products']=[r for p in parts for r in p['products']]
    result['batches']=[r for p in parts for r in p['batches']]
    for field in ('model_load_seconds','warmup_seconds','function_body_seconds'):
        result[field]=sum(p[field] for p in parts)
    result['request_products']=min(limit,2048)
    result['timing_note']='Single remote request for pilot; model load and warmup recorded separately. Includes client request overhead in remote-call wall time.'
    result['remote_call_wall_seconds'] = time.perf_counter()-start
    products = result.pop('products')
    assert len(products) == limit and {r['product_id'] for r in products} == {r['product_id'] for r in selected}
    (out/'products.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in products))
    result['count'] = limit
    seconds = sum(r['wall_seconds'] for r in result['batches'])
    result['batch_processing_seconds'] = seconds
    result['products_per_second'] = limit/seconds
    result['projected_full_batch_hours'] = 164900*seconds/limit/3600
    result['pricing_checked_date'] = '2026-10-06'
    result['pricing_source'] = 'https://modal.com/pricing'
    result['configured_rates_usd_per_second'] = {'T4': .000164, 'cpu_per_core': .0000131, 'memory_per_gib': .00000222}
    # Conservative requested-resource calculation; billed allocations and all charges must be verified in Modal usage.
    rate = .000164 + 2*.0000131 + 8*.00000222
    result['estimated_function_compute_usd'] = result['function_body_seconds']*rate
    result['estimated_full_batch_compute_usd'] = 164900*seconds/limit*rate
    result['actual_billed_usd'] = None
    result['cost_note'] = 'Pricing-based estimates, not actual billing. Function-body estimate excludes container startup/image build and tail; verify account usage before extending the run. Full-corpus extrapolation assumes this fixed metadata-only hash sample throughput holds.'
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT/f'reports/modal_t5_pilot_{limit}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'batches'}, indent=2))
