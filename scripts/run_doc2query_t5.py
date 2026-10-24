"""Pinned, resumable docT5query inference with measured local compute usage."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'castorini/doc2query-t5-base-msmarco'
REVISION = '53926b5364548f0c81b39a31723184d3664aaf06'
INPUT = ROOT / 'outputs/budget_enrichment/test/products_enriched.jsonl'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main(args):
    import psutil
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    rows = [json.loads(line) for line in INPUT.open() if line.strip()]
    if len(rows) != 164900 or len({r['product_id'] for r in rows}) != 164900:
        raise ValueError('Unexpected source coverage')
    # Metadata-only deterministic selection for throughput measurement; no test metrics.
    rows.sort(key=lambda r: hashlib.sha256(('t5-throughput-42:' + r['product_id']).encode()).digest())
    if args.limit:
        rows = rows[:args.limit]
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {'model': MODEL, 'revision': REVISION, 'source_sha256': digest(INPUT),
              'input': 'Exact saved shortened metadata supplied to structured LLM; field values concatenated.',
              'input_max_tokens': 512, 'num_return_sequences': args.queries,
              'max_length': 64, 'do_sample': True, 'top_k': 10, 'top_p': 1.0,
              'padding_policy': 'Batch maximum rounded to a multiple of 128; attention mask preserved.',
              'temperature': 1.0, 'seed': 42, 'seed_policy': 'Seed 42 + batch position; same batch layout required on resume.',
              'batch_size': args.batch_size, 'device': device, 'threads': args.threads,
              'limit': args.limit, 'torch': torch.__version__, 'transformers': transformers.__version__,
              'platform': platform.platform(), 'processor': platform.machine(),
              'cash_api_cost_usd': 0, 'electricity_cost_usd': None, 'hardware_cost_usd': None,
              'note': 'Zero API bill is not zero total cost. Power and cloud-equivalent prices are not measured.'}
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Resume configuration mismatch; choose a new output directory')
    manifest.write_text(json.dumps(config, indent=2) + '\n')
    records = out / 'products.jsonl'
    done = {}
    if records.exists():
        for line in records.open():
            row = json.loads(line)
            if row['product_id'] in done:
                raise ValueError('Duplicate saved product')
            done[row['product_id']] = row
    selected = {r['product_id'] for r in rows}
    if set(done) - selected:
        raise ValueError('Unexpected saved product')
    print('Loading pinned model on', device, flush=True)
    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    load_seconds = time.perf_counter() - start_load
    process = psutil.Process()

    def synchronize():
        if device == 'mps':
            torch.mps.synchronize()
        elif device.startswith('cuda'):
            torch.cuda.synchronize()

    def infer(batch, seed):
        synchronize()
        wall = time.perf_counter()
        cpu = time.process_time()
        texts = ['\n'.join(str(v) for v in row['input'].values() if v) for row in batch]
        encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                            max_length=512, pad_to_multiple_of=128).to(device)
        torch.manual_seed(seed)
        synchronize()
        with torch.inference_mode():
            outputs = model.generate(**encoded, do_sample=True, top_k=10, top_p=1.0,
                                     temperature=1.0, max_length=64, num_return_sequences=args.queries)
        synchronize()
        output_cpu = outputs.cpu()
        decoded = tokenizer.batch_decode(output_cpu, skip_special_tokens=True)
        seconds = time.perf_counter() - wall
        cpu_seconds = time.process_time() - cpu
        return encoded, output_cpu, decoded, seconds, cpu_seconds

    # Separate warm-up from measured inference; never append the warm-up predictions.
    warm_start = time.perf_counter()
    infer(rows[:args.batch_size], 42)
    warmup_seconds = time.perf_counter() - warm_start
    print('Warm-up complete; generating', len(rows) - len(done), 'products', flush=True)
    with records.open('a') as stream, (out / 'batches.jsonl').open('a') as batch_stream:
        for position in range(0, len(rows), args.batch_size):
            original_batch = rows[position:position + args.batch_size]
            saved = [r['product_id'] in done for r in original_batch]
            if all(saved):
                continue
            # Regenerate a partially written batch under the same seed, append only missing rows.
            encoded, outputs, decoded, seconds, cpu_seconds = infer(original_batch, 42 + position)
            pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
            batch_row = {'position': position, 'products_processed': len(original_batch),
                         'new_products': sum(not x for x in saved), 'wall_seconds': seconds,
                         'process_cpu_seconds': cpu_seconds, 'process_rss_bytes_after_batch': process.memory_info().rss,
                         'input_tokens': int(encoded['attention_mask'].sum().item()),
                         'generated_non_special_tokens': int(((outputs != pad) & (outputs != eos)).sum().item())}
            # Retain compute for repeated batches even if a run is interrupted while writing products.
            batch_stream.write(json.dumps(batch_row) + '\n'); batch_stream.flush()
            for i, row in enumerate(original_batch):
                if saved[i]:
                    continue
                result = {'product_id': row['product_id'],
                          'queries': decoded[i * args.queries:(i + 1) * args.queries]}
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                done[row['product_id']] = result
            stream.flush()
            if device == 'mps': torch.mps.empty_cache()
            print(f'{len(done)}/{len(rows)} products; batch {seconds:.2f}s', flush=True)
    batches = [json.loads(line) for line in (out / 'batches.jsonl').open()]
    total_seconds = sum(r['wall_seconds'] for r in batches)
    processed = sum(r['products_processed'] for r in batches)
    summary = {'complete': set(done) == selected, 'products': len(done), 'selected_products': len(rows),
               'generation_wall_seconds': total_seconds,
               'timing_scope': 'Batch input construction, tokenization, device transfers, generation, synchronization and decoding; excludes file writes and model loading.',
               'process_cpu_seconds': sum(r['process_cpu_seconds'] for r in batches),
               'products_per_second': processed / total_seconds,
               'largest_observed_process_rss_bytes': max(r['process_rss_bytes_after_batch'] for r in batches),
               'model_load_seconds_latest_session': load_seconds, 'warmup_seconds_latest_session': warmup_seconds,
               'projected_164900_generation_hours': 164900 * total_seconds / processed / 3600,
               'projection_note': 'Throughput extrapolation, not a measured full-run time; excludes setup, idle time and thermal changes.',
               'cash_api_cost_usd': 0, 'electricity_cost_usd': None,
               'input_tokens': sum(r['input_tokens'] for r in batches),
               'generated_non_special_tokens': sum(r['generated_non_special_tokens'] for r in batches),
               'products_sha256': digest(records), 'configuration': config}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'sessions.jsonl').open('a') as stream:
        stream.write(json.dumps({'model_load_seconds': load_seconds, 'warmup_seconds': warmup_seconds}) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='outputs/doc2query_t5/throughput')
    p.add_argument('--limit', type=int, default=32, help='0 for all 164,900 products; throughput sample otherwise')
    p.add_argument('--queries', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    args = p.parse_args()
    if args.limit < 0 or args.limit > 164900 or min(args.queries, args.batch_size, args.threads) < 1:
        p.error('Invalid limits')
    main(args)
