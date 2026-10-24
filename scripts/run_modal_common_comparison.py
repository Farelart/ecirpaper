"""Wait for existing E5 call, then launch exactly one common-host benchmark."""
import datetime,json,subprocess,time
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
CLI='/Users/farelganlaky/.local/bin/modal'
OUT=ROOT/'outputs/modal_common_t4'
STATUS=ROOT/'reports/modal_common_t4_status.json'

def update(stage,detail):
    s=json.loads(STATUS.read_text());s['stage']=stage;s['detail']=detail
    s['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    STATUS.write_text(json.dumps(s,indent=2)+'\n');print(stage,detail,flush=True)

def get(volume,remote,local):
    return subprocess.run([CLI,'volume','get',volume,remote,str(local),'--force'],
                          capture_output=True,text=True)

def main():
    started=time.monotonic()
    while True:
        result=get('ecir-common-t4-benchmark','/inputs_manifest.json',OUT/'uploaded_inputs_manifest.json')
        if result.returncode==0:
            assert json.loads((OUT/'uploaded_inputs_manifest.json').read_text())==json.loads((OUT/'inputs_manifest.json').read_text())
            break
        if time.monotonic()-started>3600:raise TimeoutError('Input upload did not complete')
        time.sleep(15)
    update('e5_encoding','Waiting for saved T4 encoding call; no duplicate inference')
    saved=json.loads((ROOT/'outputs/modal_e5_full/call.json').read_text())
    call=modal.FunctionCall.from_id(saved['call_id'])
    while True:
        try:
            result=call.get(timeout=30);summary=json.loads(result);break
        except TimeoutError:
            get(saved['volume'],'/e5_t4/progress.json',ROOT/'outputs/modal_e5_full/progress.json')
            if time.monotonic()-started>18000:raise TimeoutError('Cloud E5 did not finish within expected bound')
    assert summary['complete'] and summary['products']==164900 and summary['queries']==8956
    (ROOT/'outputs/modal_e5_full/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    update('common_benchmark','E5 T4 complete; launching one sequential all-method benchmark')
    if not (OUT/'call.json').exists():
        with (OUT/'run.log').open('a') as log:
            subprocess.run([CLI,'run','--detach','scripts/modal_common_t4_benchmark.py'],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    saved=json.loads((OUT/'call.json').read_text())
    call=modal.FunctionCall.from_id(saved['call_id'])
    while True:
        try:summary=json.loads(call.get(timeout=30));break
        except TimeoutError:
            get(saved['volume'],'/results/progress.json',OUT/'progress.json')
            if time.monotonic()-started>24000:raise TimeoutError('Common benchmark exceeded bound')
    assert summary['complete'] and len(summary['methods'])==8
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    update('download','Eight method benchmarks complete; retrieving saved rankings/timings')
    dest=OUT/'checkpoints';dest.mkdir(exist_ok=True)
    result=get(saved['volume'],'/results/',dest)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    update('validation','Running independent cloud-ranking and timing validation')
    subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/verify_modal_common_results.py'],cwd=ROOT,check=True)
    for kind,args in [('report',['billing','report','--start','2026-10-05','--resolution','h','--tz','Africa/Porto-Novo','--show-resources','--json']),
                      ('summary',['billing','summary','--for','2026-10','--json'])]:
        with (OUT/f'billing_{kind}.json').open('w') as f:
            subprocess.run([CLI,*args],stdout=f,check=True)
    update('ready_for_report','Cloud rankings/timings independently verified; billing downloaded; synthesize updated comparison')

if __name__=='__main__':
    try:main()
    except Exception as e:
        update('error',type(e).__name__+': '+str(e));raise
