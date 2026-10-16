"""Streamlit analytics dashboard. Requires reproducibly generated parquet artifacts."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Customer Retention Intelligence',page_icon='📊',layout='wide')
DATA = Path(__file__).parent/'data/prepared'

@st.cache_data
def load():
    if not (DATA/'metadata.json').exists():
        return None
    names = ['orders','customers','cohorts','risk','model_scores']
    tables = {name:pd.read_parquet(DATA/f'{name}.parquet') for name in names}
    tables['metadata'] = json.loads((DATA/'metadata.json').read_text())
    tables['orders']['InvoiceDate'] = pd.to_datetime(tables['orders'].InvoiceDate)
    return tables

def chart(fig):
    fig.update_layout(margin=dict(t=48,b=20,l=10,r=10))
    st.plotly_chart(fig,use_container_width=True)

st.title('Customer Segmentation & Retention Intelligence')
st.caption('UCI Online Retail | British pounds | historical descriptive analysis')
d = load()
if d is None:
    st.error('Prepared artifacts are missing. Download the UCI Online Retail Excel file, then run python -m src.pipeline. Deploy the generated data/prepared files alongside this app.')
    st.stop()

orders,customers,cohorts,risk = d['orders'],d['customers'],d['cohorts'],d['risk']
page = st.sidebar.radio('Dashboard',['Executive Overview','Customer Segments','Retention Analysis','Churn Risk','Data Explorer'])
min_date,max_date = orders.InvoiceDate.min().date(),orders.InvoiceDate.max().date()
dates = st.sidebar.date_input('Order dates',(min_date,max_date),min_value=min_date,max_value=max_date)
segments = st.sidebar.multiselect('Cutoff-date RFM segments',sorted(customers.Segment.unique()),default=sorted(customers.Segment.unique()))
if not isinstance(dates,tuple) or len(dates)!=2:
    st.info('Select a start and end date.'); st.stop()
start,end = map(pd.Timestamp,dates)
selected_customers = customers[customers.Segment.isin(segments)]
filtered = orders[(orders.InvoiceDate>=start)&(orders.InvoiceDate<end+pd.Timedelta(days=1))&
                  (orders.CustomerID.isin(selected_customers.CustomerID))]
st.sidebar.caption('Analysis cutoff: '+str(d['metadata']['cutoff']))
if filtered.empty:
    st.info('No purchases match these filters. Change the dates or segments.'); st.stop()

if page == 'Executive Overview':
    count = filtered.CustomerID.nunique()
    repeat = (filtered.groupby('CustomerID').InvoiceNo.nunique()>=2).sum()/count
    metrics = [('Customers',f'{count:,}'),('Orders',f'{filtered.InvoiceNo.nunique():,}'),
               ('Accepted gross sales',f'£{filtered.Revenue.sum():,.0f}'),('Repeat purchase share',f'{repeat:.1%}')]
    for col,(label,value) in zip(st.columns(4),metrics):
        col.metric(label,value)
    st.caption('All KPIs use selected purchases and customers in cutoff-date segments. Repeat purchase share: customers with 2+ distinct orders / customers with 1+ orders, within the selected range. Gross sales exclude returns and cancellations; they are not refund-adjusted net sales.')
    by_month = filtered.assign(month=filtered.InvoiceDate.dt.to_period('M').astype(str)).groupby('month',as_index=False).Revenue.sum()
    chart(px.line(by_month,x='month',y='Revenue',markers=True,title='Monthly accepted gross sales'))
    mix = filtered.merge(customers[['CustomerID','Segment']],on='CustomerID').groupby('Segment',as_index=False).Revenue.sum()
    chart(px.bar(mix,x='Segment',y='Revenue',title='Sales by full-history customer segment'))
elif page == 'Customer Segments':
    st.subheader('Customer segments')
    view = selected_customers[selected_customers.CustomerID.isin(filtered.CustomerID)]
    grouped = view.groupby('Segment').agg(Customers=('CustomerID','nunique'),TotalSpend=('Monetary','sum'),
             AverageSpend=('Monetary','mean'),AverageOrders=('Frequency','mean')).reset_index()
    grouped['RevenueShare'] = grouped.TotalSpend/grouped.TotalSpend.sum()
    st.caption('RFM scores use all eligible history through the fixed cutoff. Filters only restrict customers displayed, not their historic RFM features or cohort definitions.')
    st.dataframe(grouped,hide_index=True,use_container_width=True)
    chart(px.bar(grouped,x='Segment',y='Customers',title='Segment distribution'))
    chart(px.bar(grouped,x='Segment',y='TotalSpend',title='Full-history sales contribution'))
    chart(px.scatter(view,x='Recency',y='Monetary',size='Frequency',color='Segment',log_y=True,
           hover_data=['CustomerID'],title='Customer recency, spend and order frequency'))
    st.info('Champions: R≥4,F≥4. At risk: R≤2,F≥3. Promising: R≥4,F≤2. Loyal: remaining F≥3,M≥3. Occasional: others. Rules evaluated in stated order. Percentile scores are relative to the observed customer population.')
    st.dataframe(view,hide_index=True,use_container_width=True)
elif page == 'Retention Analysis':
    st.subheader('Monthly repeat-purchase retention')
    st.caption('Cohort = month of first observed order. Retention in month N = distinct cohort customers who bought in month N divided by cohort size, not cumulative repeat purchasing. Fixed full-history cohorts ignore dashboard filters. Most recent cohorts are not fully observed; customers before dataset start are left-censored.')
    pivot = cohorts.pivot(index='cohort',columns='period',values='retention_pct')
    chart(px.imshow(pivot,aspect='auto',title='Monthly cohort retention (%)',
         labels={'x':'Months after observed acquisition','y':'First observed month','color':'Retention %'}))
    chart(px.line(cohorts[cohorts.period>0],x='period',y='retention_pct',color='cohort',markers=True,title='Retention curves by cohort'))
    st.dataframe(cohorts,hide_index=True,use_container_width=True)
elif page == 'Churn Risk':
    st.subheader('Predicted 90-day customer inactivity')
    st.caption('A positive label means no order in the next 90 days. This is not contractual churn. Historical model test scores are separate from current predictions.')
    st.info(d['metadata']['model_note'])
    scores = d['model_scores']
    if scores.empty or 'risk' not in risk.columns:
        st.warning('No defensible evaluated model is available from this observation window.')
    else:
        st.dataframe(scores,hide_index=True,use_container_width=True)
        frame = risk[risk.Segment.isin(segments)].dropna(subset=['risk'])
        chart(px.histogram(frame,x='risk',nbins=20,title='Estimated probability of inactivity'))
        chart(px.bar(frame.groupby('Segment',as_index=False).risk.mean(),x='Segment',y='risk',title='Average risk by segment'))
        st.dataframe(frame.sort_values('risk',ascending=False),hide_index=True,use_container_width=True)
        st.caption('Feature importance omitted; no validated global importance calculation is available.')
else:
    st.subheader('Data explorer')
    choice = st.selectbox('Data table',['Orders','Customers','Cohorts','Predicted risk'])
    tables = {'Orders':filtered,'Customers':selected_customers[selected_customers.CustomerID.isin(filtered.CustomerID)],
              'Cohorts':cohorts,'Predicted risk':risk[risk.Segment.isin(segments)]}
    table = tables[choice]
    st.dataframe(table,hide_index=True,use_container_width=True)
    st.download_button('Export selected table (CSV)',table.to_csv(index=False),file_name=choice.lower().replace(' ','_')+'.csv',mime='text/csv')
st.divider()
st.caption('Data: UCI Online Retail, CC BY 4.0 (DOI: 10.24432/C5BW33). Observation begins Dec 2010; historic customer activity before then is unknown.')
