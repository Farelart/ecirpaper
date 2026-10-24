"""Independently check complete E5 T4 checkpoint coverage and representations."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    base=ROOT/'outputs/modal_e5_full/checkpoints/e5_t4'
    summary=json.loads((base/'summary.json').read_text());cfg=summary['configuration']
    assert summary['complete']
    assert cfg['model']=='intfloat/e5-small-v2' and cfg['revision']=='ffb93f3bd4047442299a41ebb6fa998a38507c52'
    assert cfg['batch_size']==32 and cfg['max_length']==512 and cfg['gpu']=='T4' and cfg['precision']=='float32'
    assert cfg['pooling']=='masked mean + L2 normalization'
    assert cfg['prefixes']=={'products':'passage: ','queries':'query: '}
    expected=json.loads((ROOT/'reports/splade_encoding_validation.json').read_text())['configuration']['input_sha256']
    assert cfg['input_sha256']==expected
    records={}
    for kind,n in [('products',164900),('queries',8956)]:
        ids=json.loads((base/f'{kind}_ids.json').read_text())
        source_ids=json.loads((ROOT/f'outputs/splade/checkpoints/encoded/{kind}_ids.json').read_text())
        assert ids==source_ids and len(ids)==len(set(ids))==n
        max_error=0;parts=0;seconds=0
        for offset in range(0,n,2048):
            p=base/f'{kind}_{offset:06d}.npy';meta=json.loads(p.with_suffix('.json').read_text())
            assert meta['ids']==ids[offset:offset+2048]
            a=np.load(p);assert a.shape==(len(meta['ids']),384) and a.dtype==np.float32
            assert np.isfinite(a).all();error=float(np.max(np.abs(np.linalg.norm(a,axis=1)-1)))
            assert error<1e-5;max_error=max(error,max_error);seconds+=meta['processing_seconds'];parts+=1
        assert abs(seconds-summary['processing_seconds'][kind])<1e-6
        records[kind]={'rows':n,'checkpoint_files':parts,'all_finite':True,'max_l2_norm_error':max_error,'processing_seconds':seconds}
    report={'passed':True,'configuration':cfg,'coverage':records,
            'summary_sha256':hashlib.sha256((base/'summary.json').read_bytes()).hexdigest(),
            'note':'Independent local verification of all checkpoint arrays, ordered unique input IDs, pinned settings and summed recorded durations.'}
    (ROOT/'reports/e5_t4_encoding_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
