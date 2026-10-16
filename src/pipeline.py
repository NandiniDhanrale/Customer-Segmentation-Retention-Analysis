"""Reproducible ETL. Run python -m src.pipeline after downloading original UCI workbook."""
from pathlib import Path
import json
import pandas as pd
from .analytics import clean_transactions,orders_from_lines,rfm,retention,snapshots
from .model import train_and_evaluate,FEATURES

def build(source='data/Online Retail.xlsx', out='data/prepared'):
    raw = pd.read_excel(source,engine='openpyxl')
    lines = clean_transactions(raw)
    orders = orders_from_lines(lines)
    if orders.empty:
        raise ValueError('No eligible orders remain after validation.')
    cutoff = orders.InvoiceDate.max()
    customers = rfm(orders,cutoff)
    cohorts = retention(orders,cutoff)
    scores,model,note = train_and_evaluate(snapshots(orders))
    risk = customers[['CustomerID','Segment']].copy()
    if model is not None:
        grouped = orders.groupby('CustomerID').agg(last=('InvoiceDate','max'),frequency=('InvoiceNo','nunique'),
                                                     monetary=('Revenue','sum'),first=('InvoiceDate','min'))
        grouped['recency'] = (cutoff-grouped['last']).dt.days
        grouped['tenure'] = (cutoff-grouped['first']).dt.days
        grouped['mean_order_value'] = grouped.monetary/grouped.frequency
        grouped['risk'] = model.predict_proba(grouped[FEATURES])[:,1]
        risk = risk.merge(grouped[['risk']],left_on='CustomerID',right_index=True,how='left')
    target = Path(out)
    target.mkdir(parents=True,exist_ok=True)
    for name,frame in [('orders',orders),('customers',customers),('cohorts',cohorts),
                       ('model_scores',scores),('risk',risk)]:
        frame.to_parquet(target/f'{name}.parquet',index=False)
    (target/'metadata.json').write_text(json.dumps({
      'cutoff':str(cutoff),'model_note':note,'raw_rows':len(raw),'clean_lines':len(lines),'orders':len(orders),
      'sales_definition':'Accepted positive-value purchase lines only. Excludes cancellations/returns; NOT refund-adjusted net sales.'
    },indent=2))
    return target

if __name__=='__main__':
    print(build())
