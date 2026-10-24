"""Complete test-product enrichment, with bounded individual retries."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credentials-file',type=Path)
    args=p.parse_args()
    command=[sys.executable,str(ROOT/'scripts/run_budget_enrichment.py'),'--scope','test','--workers','32']
    if args.credentials_file:command+=['--credentials-file',str(args.credentials_file)]
    # Initial batches, then at most two individual attempts for unresolved items.
    for phase in range(3):
        current=command if phase==0 else command+['--retry-failed','--batch-size','1']
        subprocess.run(current,cwd=ROOT,check=True)
        path=ROOT/'outputs/budget_enrichment/test/products_enriched.jsonl'
        count=sum(1 for line in path.open() if line.strip()) if path.exists() else 0
        summary=json.loads((ROOT/'outputs/budget_enrichment/test/summary.json').read_text())
        print(json.dumps({'phase':phase,'enriched_products':count,'target':164900},indent=2),flush=True)
        if count==164900: return
        if summary['stopped_for_safety']:raise SystemExit('Stopped at budget, price or request safety check. Preserve ledger; investigate before resuming.')
    raise SystemExit('Some products remain unresolved after bounded retries; output is incomplete.')

if __name__=='__main__':main()
