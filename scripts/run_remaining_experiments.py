"""Run full remaining comparisons sequentially and preserve restart checkpoints."""
import json
import argparse
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv-neural/bin/python'
STATUS=ROOT/'reports/remaining_experiments_status.json'

def main(wait_for_encoder_pid=None):
    stages=[('encoding','run_e5_embeddings.py',[],ROOT/'outputs/e5_small/summary.json'),
            ('dense_and_hybrid_evaluation','evaluate_neural_methods.py',[],ROOT/'reports/neural_evaluation.json'),
            ('t5_generation','run_doc2query_t5.py',['--limit','0','--batch-size','16','--output','outputs/doc2query_t5/full_b16'],ROOT/'outputs/doc2query_t5/full_b16/summary.json'),
            ('t5_evaluation','evaluate_doc2query_t5.py',[],ROOT/'reports/doc2query_evaluation.json')]
    state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'complete':False,'stages':{}}
    out=ROOT/'outputs/remaining_experiments';out.mkdir(parents=True,exist_ok=True)
    if wait_for_encoder_pid:
        state['current_stage']='encoding'
        state['stages']['encoding']={'state':'running','existing_pid':wait_for_encoder_pid}
        STATUS.write_text(json.dumps(state,indent=2)+'\n')
        print('Waiting for the already-running encoder:',wait_for_encoder_pid,flush=True)
        artifact=ROOT/'outputs/e5_small/summary.json'
        while True:
            if artifact.exists() and json.loads(artifact.read_text()).get('complete'):break
            try:os.kill(wait_for_encoder_pid,0)
            except ProcessLookupError:raise SystemExit('Existing encoder exited before completion; inspect its log')
            time.sleep(5)
    for name,script,args,artifact in stages:
        if artifact.exists():
            saved=json.loads(artifact.read_text())
            if saved.get('complete') or name.endswith('evaluation'):
                state['stages'][name]={'state':'complete','artifact':str(artifact)}
                print('Validated stage artifact exists:',name,flush=True);continue
        state['current_stage']=name;state['stages'][name]={'state':'running','artifact':str(artifact)}
        STATUS.write_text(json.dumps(state,indent=2)+'\n')
        print('Starting',name,flush=True);start=time.perf_counter()
        with (out/f'{name}.log').open('a') as stream:
            result=subprocess.run([str(PY),'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        state['stages'][name]['session_wall_seconds']=time.perf_counter()-start
        if result.returncode:
            state['stages'][name]['state']='failed';state['stages'][name]['returncode']=result.returncode
            STATUS.write_text(json.dumps(state,indent=2)+'\n');raise SystemExit(f'{name} failed; outputs and logs preserved')
        if not artifact.exists():raise ValueError('Stage produced no completion artifact')
        state['stages'][name]['state']='complete'
        STATUS.write_text(json.dumps(state,indent=2)+'\n')
        print('Completed',name,flush=True)
    state['complete']=True;state['current_stage']=None;state['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    STATUS.write_text(json.dumps(state,indent=2)+'\n');print('All remaining full comparisons complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-for-encoder-pid',type=int)
    main(p.parse_args().wait_for_encoder_pid)
