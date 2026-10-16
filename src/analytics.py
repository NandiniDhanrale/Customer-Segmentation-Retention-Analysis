"""Validated purchase analytics. All money is in source GBP."""
import numpy as np
import pandas as pd

REQUIRED = {'InvoiceNo', 'InvoiceDate', 'CustomerID', 'Quantity', 'UnitPrice', 'StockCode'}

def clean_transactions(raw):
    absent = REQUIRED - set(raw.columns)
    if absent:
        raise ValueError(f'Missing columns: {sorted(absent)}')
    df = raw.copy().drop_duplicates()
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
    for col in ('Quantity', 'UnitPrice'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['InvoiceNo', 'CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice'])
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
    df = df[~df.InvoiceNo.astype(str).str.upper().str.startswith('C')]
    df['CustomerID'] = df.CustomerID.astype('Int64').astype(str)
    df['InvoiceNo'] = df.InvoiceNo.astype(str)
    df['Revenue'] = df.Quantity * df.UnitPrice
    return df

def orders_from_lines(lines):
    import duckdb
    with duckdb.connect(':memory:') as con:
        con.register('lines', lines)
        return con.execute("""
          SELECT CustomerID, InvoiceNo, MIN(InvoiceDate) AS InvoiceDate,
                 SUM(Revenue) AS Revenue, SUM(Quantity) AS Items
          FROM lines GROUP BY CustomerID, InvoiceNo
        """).df()

def rfm(orders, cutoff):
    cutoff = pd.Timestamp(cutoff)
    past = orders[orders.InvoiceDate <= cutoff]
    if past.empty:
        return pd.DataFrame(columns=['CustomerID','Recency','Frequency','Monetary','R','F','M','Segment'])
    df = past.groupby('CustomerID').agg(last=('InvoiceDate','max'), Frequency=('InvoiceNo','nunique'), Monetary=('Revenue','sum')).reset_index()
    df['Recency'] = (cutoff - df['last']).dt.days
    df['R'] = 6 - np.ceil(df.Recency.rank(pct=True, method='average') * 5).clip(1,5).astype(int)
    df['F'] = np.ceil(df.Frequency.rank(pct=True, method='average') * 5).clip(1,5).astype(int)
    df['M'] = np.ceil(df.Monetary.rank(pct=True, method='average') * 5).clip(1,5).astype(int)
    rules = [(df.R >= 4) & (df.F >= 4), (df.R <= 2) & (df.F >= 3), (df.R >= 4) & (df.F <= 2), (df.F >= 3) & (df.M >= 3)]
    df['Segment'] = np.select(rules, ['Champions','At risk','Promising','Loyal'], default='Occasional')
    return df.drop(columns='last')

def retention(orders, cutoff):
    x = orders[orders.InvoiceDate <= pd.Timestamp(cutoff)].copy()
    if x.empty:
        return pd.DataFrame(columns=['cohort','period','customers','cohort_size','retention_pct'])
    x['month'] = x.InvoiceDate.dt.to_period('M')
    first = x.groupby('CustomerID').month.min().rename('cohort')
    x = x.join(first, on='CustomerID')
    x['period'] = (x.month.dt.year-x.cohort.dt.year)*12 + x.month.dt.month-x.cohort.dt.month
    actual = x.groupby(['cohort','period']).CustomerID.nunique().rename('customers').reset_index()
    sizes = x.groupby('cohort').CustomerID.nunique().rename('cohort_size')
    final_month = pd.Timestamp(cutoff).to_period('M')
    cells = [(c,p) for c in sizes.index for p in range((final_month.year-c.year)*12+final_month.month-c.month+1)]
    full = pd.DataFrame(cells, columns=['cohort','period']).merge(actual, how='left', on=['cohort','period'])
    full['customers'] = full.customers.fillna(0).astype(int)
    full = full.join(sizes, on='cohort')
    full['retention_pct'] = 100*full.customers/full.cohort_size
    full['cohort'] = full.cohort.astype(str)
    return full

def snapshots(orders, horizon_days=90, step_days=30, min_history_days=30):
    """Label 1 means no purchase in (snapshot, snapshot + horizon]; prior features only."""
    x = orders.sort_values('InvoiceDate')
    if x.empty:
        return pd.DataFrame()
    start = x.InvoiceDate.min()+pd.Timedelta(days=min_history_days)
    end = x.InvoiceDate.max()-pd.Timedelta(days=horizon_days)
    results = []
    for at in pd.date_range(start.normalize(), end.normalize(), freq=f'{step_days}D'):
        past = x[x.InvoiceDate <= at]
        upcoming = x[(x.InvoiceDate > at) & (x.InvoiceDate <= at+pd.Timedelta(days=horizon_days))]
        if past.empty:
            continue
        agg = past.groupby('CustomerID').agg(last=('InvoiceDate','max'), frequency=('InvoiceNo','nunique'), monetary=('Revenue','sum'), first=('InvoiceDate','min'))
        agg['recency'] = (at-agg['last']).dt.days
        agg['tenure'] = (at-agg['first']).dt.days
        agg['mean_order_value'] = agg.monetary/agg.frequency
        agg['inactive_90d'] = (~agg.index.isin(upcoming.CustomerID)).astype(int)
        agg['snapshot'] = at
        results.append(agg.reset_index()[['CustomerID','snapshot','recency','frequency','monetary','tenure','mean_order_value','inactive_90d']])
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
