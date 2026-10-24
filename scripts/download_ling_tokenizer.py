"""Download only the pinned official Ling tokenizer, not model weights."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REVISION='ef06d91fe382109ae82647da88ff99b0f11745b0'
out=ROOT/'data/models/ling_tokenizer'
out.mkdir(parents=True,exist_ok=True)
files=[]
for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
    url=f'https://huggingface.co/inclusionAI/Ling-3.0-flash/resolve/{REVISION}/{name}'
    path=out/name
    if not path.exists():
        partial=path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url,partial)
        partial.replace(path)
    files.append({'name':name,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(out/'revision.txt').write_text(REVISION+'\n')
(out/'manifest.json').write_text(json.dumps({'revision':REVISION,'files':files},indent=2)+'\n')
print('Pinned tokenizer ready; no model weights downloaded')
