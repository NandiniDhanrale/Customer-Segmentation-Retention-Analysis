import pandas as pd
import pytest
from src.analytics import clean_transactions,rfm,retention,snapshots

def test_cleaning_excludes_cancellations_returns_and_duplicates():
    raw = pd.DataFrame({'InvoiceNo':['1','1','C2','3','4'],'InvoiceDate':['2021-01-01']*5,
      'CustomerID':[1,1,1,1,None],'Quantity':[2,2,-1,1,2],
      'UnitPrice':[5,5,5,0,7],'StockCode':['A']*5})
    result = clean_transactions(raw)
    assert len(result)==1
    assert result.Revenue.iloc[0]==10

def sample_orders():
    return pd.DataFrame({'CustomerID':['1','1','2','1'],
      'InvoiceNo':['a','b','c','d'],
      'InvoiceDate':pd.to_datetime(['2021-01-01','2021-02-01','2021-01-05','2021-04-01']),
      'Revenue':[10.,20.,30.,15.]})

def test_rfm_values():
    result = rfm(sample_orders(),pd.Timestamp('2021-04-30')).set_index('CustomerID')
    assert result.loc['1','Frequency']==3
    assert result.loc['1','Monetary']==45
    assert result.loc['1','Recency']==29

def test_cohorts_not_cumulative():
    result = retention(sample_orders(),pd.Timestamp('2021-04-30'))
    jan = result[result.cohort=='2021-01'].set_index('period')
    assert jan.loc[0,'cohort_size']==2
    assert jan.loc[1,'customers']==1
    assert jan.loc[2,'customers']==0
    assert jan.loc[3,'customers']==1

def test_no_future_features():
    x = snapshots(sample_orders(),horizon_days=30,step_days=10,min_history_days=10)
    if not x.empty:
        assert (x.recency>=0).all()
        for row in x.itertuples():
            previous = sample_orders()[(sample_orders().CustomerID==row.CustomerID)&
                                       (sample_orders().InvoiceDate<=row.snapshot)]
            assert row.frequency==previous.InvoiceNo.nunique()

def test_missing_required_columns():
    with pytest.raises(ValueError):
        clean_transactions(pd.DataFrame({'CustomerID':[1]}))
