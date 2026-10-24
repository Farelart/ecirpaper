"""Wait for checkpointed generation, then evaluate and benchmark sequentially."""
import json,subprocess,time,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'reports/final_experiments_status.json'
MODAL='/Users/farelganlaky/.local/bin/modal'
def update(stage,detail):
    s=json.loads(STATUS.read_text());s['stages'][stage]=detail;STATUS.write_text(json.dumps(s,indent=2)+'\n');print(stage,detail,flush=True)
def main():
    started=time.monotonic()
    ling=ROOT/'outputs/budget_enrichment/ling_doc2query'
    while True:
        summary=json.loads((ling/'summary.json').read_text()) if (ling/'summary.json').exists() else {}
        if summary.get('complete'):break
        process=json.loads((ling/'process.json').read_text())
        try:os.kill(process['pid'],0)
        except ProcessLookupError:raise RuntimeError('Ling control stopped incomplete; inspect log before recovery')
        if time.monotonic()-started>18000:raise TimeoutError('Ling control wait exceeded five hours')
        time.sleep(30)
    update('ling_doc2query','complete; 164900 products; three queries each')
    out=ROOT/'outputs/splade';check=out/'checkpoints';check.mkdir(exist_ok=True)
    while True:
        if (out/'summary.json').exists():
            try:summary=json.loads((out/'summary.json').read_text())
            except ValueError:summary={}
            if summary.get('complete'):break
        result=subprocess.run([MODAL,'volume','get','ecir-splade-benchmark','/encoded/summary.json',str(out/'summary.json'),'--force'],capture_output=True,text=True)
        if result.returncode==0:
            try:summary=json.loads((out/'summary.json').read_text())
            except (ValueError,FileNotFoundError):summary={}
            if summary.get('complete'):break
        if time.monotonic()-started>21600:raise TimeoutError('SPLADE wait exceeded six hours; inspect cloud status before recovery')
        time.sleep(60)
    local=check/'encoded'
    complete_local=(local/'summary.json').exists() and (local/'manifest.json').exists()
    for kind,n in [('products',164900),('queries',8956)]:
        for offset in range(0,n,2048):
            complete_local=complete_local and (local/f'{kind}_{offset:06d}.json').exists() and (local/f'{kind}_{offset:06d}.npz').exists() and (local/f'{kind}_{offset:06d}.npz').stat().st_size>0
    if complete_local:
        complete_local=json.loads((local/'summary.json').read_text())==summary
    if not complete_local:
        subprocess.run([MODAL,'volume','get','ecir-splade-benchmark','/encoded/',str(check),'--force'],check=True)
    update('splade','encoding complete; checkpoints downloaded; candidate quality verification pending')
    update('indexing_retrieval','running sequentially; all 8956 queries repeated three times for each method')
    subprocess.run([str(ROOT/'.venv-neural/bin/python'),'-u',str(ROOT/'scripts/benchmark_final_methods.py')],cwd=ROOT,check=True)
    update('indexing_retrieval','complete; reports/retrieval_benchmark.json')
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/verify_rankings_sql.py')],cwd=ROOT,check=True)
    update('independent_quality_validation','complete; reports/ranking_sql_validation_final.json')
    update('paper_update','Ready for verified-result synthesis; do not claim whole-catalog recall or network serving latency')
    s=json.loads(STATUS.read_text());s['benchmark_pipeline_complete']=True;STATUS.write_text(json.dumps(s,indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except Exception as e:
        update('pipeline_error',type(e).__name__+': '+str(e));raise
