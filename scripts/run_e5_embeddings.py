"""Encode all fixed ESCI test products and queries with pinned E5-small."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'intfloat/e5-small-v2'
REVISION = 'ffb93f3bd4047442299a41ebb6fa998a38507c52'
DATA = ROOT / 'outputs/neural_inputs'

def main(args):
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    out = ROOT / 'outputs/e5_small'; out.mkdir(parents=True, exist_ok=True)
    source = json.loads((DATA / 'manifest.json').read_text())
    config = {'model': MODEL, 'revision': REVISION, 'device': device, 'precision': 'float32',
              'dimension': 384, 'max_tokens': 512, 'batch_size': args.batch_size,
              'pooling': 'attention-mask mean, followed by L2 normalization',
              'torch': torch.__version__, 'platform': platform.platform(), 'sources': source['files']}
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config: raise ValueError('Configuration changed')
    manifest.write_text(json.dumps(config, indent=2) + '\n')
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True, use_safetensors=True).to(device).eval()
    load_seconds = time.perf_counter() - start
    process = psutil.Process()
    def sync():
        if device == 'mps': torch.mps.synchronize()
    summary = {'configuration': config, 'model_load_seconds': load_seconds, 'api_charges_usd': 0,
               'electricity_and_hardware_cost_usd': None, 'stages': {}}
    for kind, prefix, id_name, text_name, expected in (
        ('products', 'passage: ', 'product_id', 'text', 164900),
        ('queries', 'query: ', 'query_id', 'query', 8956)):
        stage_start = time.perf_counter(); cpu_start = time.process_time()
        records = [json.loads(line) for line in (DATA / f'{kind}.jsonl').open()]
        if len(records) != expected or len({r[id_name] for r in records}) != expected: raise ValueError('Input coverage mismatch')
        ids = [r[id_name] for r in records]
        ids_path = out / f'{kind}_ids.json'
        if ids_path.exists() and json.loads(ids_path.read_text()) != ids: raise ValueError('ID order changed')
        ids_path.write_text(json.dumps(ids) + '\n')
        token_path, len_path = out / f'{kind}_tokens.npy', out / f'{kind}_lengths.npy'
        token_meta = out / f'{kind}_tokenization.json'
        if token_meta.exists():
            token_summary = json.loads(token_meta.read_text())
            tokens = np.load(token_path, mmap_mode='r'); lengths = np.load(len_path, mmap_mode='r')
        else:
            tokens = np.lib.format.open_memmap(token_path, mode='w+', dtype=np.int32, shape=(expected, 512))
            lengths = np.lib.format.open_memmap(len_path, mode='w+', dtype=np.int32, shape=(expected,))
            tokens[:] = tokenizer.pad_token_id
            tok_start = time.perf_counter(); truncated = 0; total_input_tokens = 0
            for begin in range(0, expected, 256):
                # Untruncated tokenization makes the actual truncation count measurable.
                encoded = tokenizer([prefix + r[text_name] for r in records[begin:begin+256]],
                                    truncation=False, padding=False, verbose=False)['input_ids']
                for offset, sequence in enumerate(encoded):
                    total_input_tokens += len(sequence)
                    if len(sequence) > 512:
                        truncated += 1; sequence = sequence[:511] + [sequence[-1]]
                    i = begin + offset; lengths[i] = len(sequence); tokens[i, :len(sequence)] = sequence
                if begin % 25600 == 0: print(f'Tokenizing {kind}: {min(begin+256,expected)}/{expected}', flush=True)
            tokens.flush(); lengths.flush()
            token_summary = {'wall_seconds': time.perf_counter() - tok_start,
                             'truncated_documents': truncated, 'original_input_tokens': total_input_tokens,
                             'encoded_input_tokens': int(lengths.sum())}
            token_meta.write_text(json.dumps(token_summary, indent=2) + '\n')
        del records
        vector_path, done_path = out / f'{kind}_vectors.npy', out / f'{kind}_done.npy'
        if done_path.exists():
            vectors = np.load(vector_path, mmap_mode='r+'); done = np.load(done_path, mmap_mode='r+')
        else:
            vectors = np.lib.format.open_memmap(vector_path, mode='w+', dtype=np.float32, shape=(expected,384))
            done = np.lib.format.open_memmap(done_path, mode='w+', dtype=np.bool_, shape=(expected,)); done[:] = False; done.flush()
        # Input-length bucketing reduces padding; model and all products remain fixed.
        order = np.argsort(lengths, kind='stable')
        def encode(indices):
            # Four fixed padding shapes prevent hundreds of cached MPS graphs
            # accumulating as a length-sorted corpus progresses.
            width = min(512, ((int(lengths[indices].max())+127)//128)*128)
            tensor = torch.from_numpy(np.array(tokens[indices, :width], dtype=np.int64)).to(device)
            mask = tensor != tokenizer.pad_token_id
            sync(); wall = time.perf_counter(); cpu = time.process_time()
            with torch.inference_mode():
                hidden = model(input_ids=tensor, attention_mask=mask).last_hidden_state
                pooled = hidden.masked_fill(~mask[..., None], 0).sum(dim=1) / mask.sum(dim=1)[:,None]
                result = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
            sync()
            if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result,axis=1),1,atol=1e-5): raise ValueError('Invalid embeddings')
            return result, time.perf_counter() - wall, time.process_time() - cpu
        warm_start = time.perf_counter(); encode(order[:args.batch_size]); warm_seconds = time.perf_counter() - warm_start
        missing = order[~done[order]]
        log = out / f'{kind}_batches.jsonl'
        encode_start = time.perf_counter(); initial_done = int(done.sum())
        with log.open('a') as stream:
            for begin in range(0,len(missing),args.batch_size):
                indices = missing[begin:begin+args.batch_size]
                result, seconds, cpu_seconds = encode(indices)
                vectors[indices] = result; vectors.flush(); done[indices] = True; done.flush()
                stream.write(json.dumps({'products':len(indices),'wall_seconds':seconds,'cpu_seconds':cpu_seconds,
                                         'rss_bytes':process.memory_info().rss,'max_input_length':int(lengths[indices].max())})+'\n'); stream.flush()
                if device == 'mps':
                    torch.mps.empty_cache()
                if begin % (args.batch_size*64) == 0:
                    finished = int(done.sum()); elapsed = time.perf_counter() - encode_start
                    speed = (finished-initial_done)/elapsed
                    eta = (expected-finished)/speed if speed else 0
                    print(f'Encoding {kind}: {finished}/{expected}; {speed:.1f}/s; estimated remaining {eta/60:.1f} min',flush=True)
        if not bool(done.all()): raise ValueError('Incomplete encoding')
        batches = [json.loads(line) for line in log.open()]
        summary['stages'][kind] = {'elements': expected, 'complete': True, 'tokenization': token_summary,
                                  'model_inference_wall_seconds': sum(b['wall_seconds'] for b in batches),
                                  'model_inference_cpu_seconds': sum(b['cpu_seconds'] for b in batches),
                                  'latest_session_stage_wall_seconds': time.perf_counter()-stage_start,
                                  'latest_session_stage_cpu_seconds': time.process_time()-cpu_start,
                                  'warmup_seconds_latest_session': warm_seconds,
                                  'largest_observed_process_rss_bytes': max(b['rss_bytes'] for b in batches),
                                  'vector_file_bytes': vector_path.stat().st_size}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f'Completed {kind}: {expected}',flush=True)
    summary['complete'] = True
    summary['note'] = 'Local API bill is zero; compute, energy, storage and serving costs must remain separate. Timings are offline batched encoding, not per-search latency.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    if args.batch_size<1:p.error('Positive batch size required')
    main(args)
