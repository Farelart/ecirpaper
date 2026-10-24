"""Recover observed generation spans without summing concurrent requests."""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def iso(seconds):
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).isoformat()


def main():
    report = {'method': 'Reconstructed from each logged UTC request start and monotonic request duration. First dispatch to latest reconstructed response includes gaps, retries and restarts; excludes setup before first request and final export after last response.',
              'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'clock_note': 'UTC start plus monotonic duration gives approximate response timestamp; no continuous whole-job monotonic timer was saved.',
              'methods': {}}
    for name, scope in [('structured', 'test'), ('ling_doc2query', 'ling_doc2query')]:
        out = ROOT / 'outputs/budget_enrichment' / scope
        path = out / 'batches.jsonl'
        records = [json.loads(line) for line in path.open()]
        saved = [json.loads(line) for line in (out/'products_enriched.jsonl').open()]
        assert len(saved) == len({r['product_id'] for r in saved}) == 164900
        intervals = []
        untimed = []
        timed_ids = set()
        starts = []
        for r in records:
            start = datetime.datetime.fromisoformat(r['created_at_utc']).timestamp()
            starts.append(start)
            duration = r.get('elapsed_seconds')
            if duration is None:
                untimed.append(r)
                continue
            assert math.isfinite(duration) and duration >= 0
            intervals.append((start, start+duration))
            timed_ids.update(r['product_ids'])
        assert {r['product_id'] for r in saved} <= timed_ids
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        first, last = min(starts), max(end for start,end in intervals)
        assert max(starts) <= last
        events = sorted([(start,1) for start,end in intervals] + [(end,-1) for start,end in intervals])
        active = peak = 0
        for _,delta in events:
            active += delta
            peak = max(peak, active)
        span = last-first
        union = sum(end-start for start,end in merged)
        index = json.loads((ROOT/f'outputs/final_benchmark/{name}/summary.json').read_text())['index']['total_indexing_seconds']
        report['methods'][name] = {
            'products': 164900, 'logged_requests': len(records), 'timed_requests': len(intervals),
            'untimed_failed_requests': dict(collections.Counter(r['status'] for r in untimed)),
            'first_request_utc': iso(first), 'latest_reconstructed_response_utc': iso(last),
            'observed_generation_span_seconds': span, 'observed_generation_span_minutes': span/60,
            'union_of_timed_request_intervals_seconds': union,
            'no_timed_request_in_flight_seconds_within_span': span-union,
            'sum_of_overlapping_request_seconds': sum(end-start for start,end in intervals),
            'peak_observed_timed_request_overlap': peak,
            'index_construction_seconds': index,
            'generation_span_plus_index_seconds': span+index,
            'batches_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'note': 'Observed deployment duration includes idle/recovery gaps. Union measures time with at least one timed request in flight, not complete active pipeline time. Failed requests without durations are not separately timed. Different observed concurrency prevents intrinsic model/strategy speed attribution.'}
    (ROOT/'reports/generation_timing_recovered.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
