"""Deterministic historical tables and standard Vega-Lite figure specifications."""
import csv
import json
from datetime import datetime,timezone
from scripts.build_pre_future_package import E1,E2,E1_REPORT_HASH,E2_REPORT_HASH
from scripts.frozen_encoder_extraction import Evidence,ROOT,json_bytes,sha256,read_bound


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def execute(output):
    e=Evidence();metrics=[];forecasts=[];deltas=[]
    keys=['accuracy','macro_f1_supported','macro_f1_fixed','balanced_accuracy','nll','brier','ece_15']
    for case,folder,digest in [('E1',E1,E1_REPORT_HASH),('E2',E2,E2_REPORT_HASH)]:
        r=e.js(folder+'/report.json',digest)
        for method in r['methods']:
            metrics.append(dict(case=case,method=method,N=r['samples'],K=r['K'],supported_classes=r['metrics'][method]['supported_classes'],
                                **{k:r['metrics'][method][k] for k in keys}))
        for label,mae in [('Q',r['forecast_mae_Q']),('QD_selected',r['forecast_mae_selected'])]:
            forecasts.append(dict(case=case,method=label,MAE=mae,beta=0 if label=='Q' else r['beta'],families=len(r['S_eval'])))
        for baseline in ('uniform','new-quality-only','sbsmi'):
            deltas.append(dict(case=case,comparison='quality-plus-drift minus '+baseline,
                **{k:r['metrics']['quality-plus-drift'][k]-r['metrics'][baseline][k] for k in keys}))
    write_csv(output/'method_metrics.csv',metrics);write_csv(output/'forecast_mae.csv',forecasts);write_csv(output/'paired_point_differences.csv',deltas)
    panels=[]
    for field,title in [('macro_f1_supported','Historical supported Macro-F1'),('nll','Historical NLL (lower is better)')]:
        panels.append(dict(data=dict(values=metrics),mark='bar',width=420,height=220,
            title=title,encoding=dict(x=dict(field='method',type='nominal',axis=dict(labelAngle=-40)),
            y=dict(field=field,type='quantitative',scale=dict(zero=True)),color=dict(field='case',type='nominal'),
            xOffset=dict(field='case'),tooltip=[dict(field=k) for k in ('case','method',field)])))
    spec={'$schema':'https://vega.github.io/schema/vega-lite/v5.json','vconcat':panels,
          'usermeta':dict(status='SPEC_ONLY_NOT_RENDERED',no_uncertainty_intervals=True,
            note='E1 development selection; E2 overlapping held-forward; no significance claim')}
    (output/'historical_metrics.vl.json').write_bytes(json_bytes(spec))
    for path,h in e.hashes.items():read_bound(ROOT,path,h)
    (output/'receipt.json').write_bytes(json_bytes(dict(status='PASS_TABLES',source_hashes=e.hashes,
        metric_rows=len(metrics),forecast_rows=len(forecasts),delta_rows=len(deltas),figure='Vega-Lite spec only; rendering dependency unavailable',future_access=False)))
    (output/'checksums.json').write_bytes(json_bytes({p.name:sha256(p.read_bytes()) for p in output.iterdir() if p.is_file()}))


def main():
    output=ROOT/'data/derived/historical_research_summary'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False);execute(output);print('Summary:',output)


if __name__=='__main__':main()
