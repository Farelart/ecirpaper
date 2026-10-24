"""Create a fixed audit packet and provider-reported cost summary."""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/enrichment_pilot'
products=[json.loads(s) for s in (OUT/'products.jsonl').read_text().split('\n') if s]
records=[json.loads(s) for s in (OUT/'generations.jsonl').read_text().split('\n') if s]
byid={r['product_id']:r for r in records}
selected=[]
for stratum in sorted({r['stratum'] for r in products}):
    group=[r for r in products if r['stratum']==stratum]
    group.sort(key=lambda r:hashlib.sha256((r['product_id']+':audit42').encode()).hexdigest())
    for p in group[:5]: selected.append({'product':p,'generation':byid.get(p['product_id'])})
(OUT/'audit_packet.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
costs=[r.get('usage',{}).get('cost') for r in records]
cost=sum(float(c) for c in costs if c is not None)
summary={'completed':len(records),'statuses':dict(Counter(r['status'] for r in records)),
         'provider_reported_cost_usd':cost,'responses_with_cost':sum(c is not None for c in costs),
         'projected_164900_products_usd':cost/len(records)*164900 if records else None,
         'projection_note':'Rough linear extrapolation from a balanced quality sample, not a representative workload or quote.',
         'audit_sample':20,'audit_selection':'five per stratum, smallest SHA256(product_id + :audit42); independent of generated content',
         'audit_ready':sum(r['generation'] is not None for r in selected)}
(ROOT/'reports/enrichment_pilot_cost.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
