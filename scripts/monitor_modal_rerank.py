"""Watch one saved call, download compact results, validate, fetch billing; never launch inference."""
import json,time,subprocess,datetime
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/modal_rerank';CLI='/Users/farelganlaky/.local/bin/modal'
STATUS=ROOT/'reports/rerank_comparison_status.json'
def state(stage,**info):
    record={'stage':stage,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**info}
    STATUS.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
def download(remote,local):
    r=subprocess.run([CLI,'volume','get','ecir-common-t4-benchmark','/rerank_results/'+remote,str(local),'--force'],capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
def main():
    saved=json.loads((OUT/'call.json').read_text());call=modal.FunctionCall.from_id(saved['call_id']);begin=time.monotonic();state('running',call_id=saved['call_id'],complete=False)
    while True:
        try:
            summary=json.loads(call.get(timeout=30));break
        except TimeoutError:
            try:
                download('progress.json',OUT/'progress.json');p=json.loads((OUT/'progress.json').read_text());state('running',call_id=saved['call_id'],progress=p,complete=False)
                with (OUT/'progress_history.jsonl').open('a') as f:f.write(json.dumps(p)+'\n')
            except RuntimeError:pass
            if time.monotonic()-begin>4200:raise TimeoutError('Monitor bound exceeded; inspect existing cloud call before recovery')
    assert summary['complete'];(OUT/'summary.json').write_text(json.dumps(summary,indent=2));state('downloading',complete=False)
    dest=OUT/'results';dest.mkdir(exist_ok=True)
    for name in ['manifest.json','summary.json','pair_scores.csv','quality.csv','timing_query_ids.json']+[m+'_latencies.csv' for m in saved['config']['methods']]:download(name,dest/name)
    state('validating',complete=False)
    subprocess.run(['python3','scripts/verify_rerank_comparison.py'],cwd=ROOT,check=True)
    for name,args in [('report',['billing','report','--start','2026-10-05','--resolution','h','--tz','Africa/Porto-Novo','--show-resources','--json']),('summary',['billing','summary','--for','2026-10','--json'])]:
        with (OUT/f'billing_{name}.json').open('w') as f:subprocess.run([CLI,*args],stdout=f,check=True)
    state('complete',complete=True,report='reports/rerank_comparison_verified.md',billing='outputs/modal_rerank/billing_report.json')
if __name__=='__main__':
    try:main()
    except Exception as error:state('error',complete=False,error=type(error).__name__+': '+str(error));raise
