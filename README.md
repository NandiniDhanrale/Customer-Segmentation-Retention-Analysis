# Customer Segmentation & Retention Intelligence

A reproducible e-commerce analytics and customer inactivity prediction project using Python, Pandas, DuckDB SQL, scikit-learn, Streamlit and Plotly.

## Source data

**[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)** — Chen, D. (2015), DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33). Licensed **CC BY 4.0**. Historical transactions from **1 Dec 2010 through 9 Dec 2011** for a UK-based non-store retailer. Original dataset contains approximately 541,909 transaction lines. Source fields: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country. Monetary values are British pounds (£).

### Business definitions and limitations

* The pipeline drops exact duplicate lines, missing identifiers, invalid timestamps, zero/nonpositive quantities or prices, and invoice numbers beginning with C (cancellations). Cancellations/returns **are not netted** against earlier purchases. Therefore reported sales are **accepted gross sales**, not audited net revenue.
* An **order** is a distinct InvoiceNo per CustomerID. Multiple product lines from one invoice are aggregated with DuckDB SQL.
* The **analysis cutoff** is the last timestamp in retained data. RFM: Recency = days since last observed order; Frequency = distinct invoices; Monetary = sum of eligible invoice line revenue. Scores use tied percentile ranks from 1 to 5; recency scoring is reversed. Segment rules are hierarchical: Champions R≥4,F≥4; At risk R≤2,F≥3; Promising R≥4,F≤2; Loyal (remaining) F≥3,M≥3; Occasional otherwise.
* **Acquisition cohort** = first observed purchase month. Month-N retention = unique customers purchasing **in exactly month N** / initial cohort size. It is **not** a cumulative repeat purchase metric. Latest cohorts are not fully observed.
* **Repeat purchase share** for an applied filter = distinct customers with at least two invoices / distinct customers with at least one invoice within the selected window.
* No observations before December 2010: first-observed purchase may **not** be actual acquisition. No customer subscription status, acquisition cost, customer contact data or verified downstream churn label. A 90-day absence of orders is an **inactivity proxy**.
* Predictions use only historical features on/before the snapshot; labels inspect the next 90 days. Chronological train/validation/test splits have 90-day temporal embargoes. If there is insufficient history or class diversity, the pipeline **does not invent** metrics or a fitted model.
* Global RFM segments, cohorts and model evaluation use the fixed data cutoff; dashboard date/segment filters apply to purchased activity and displayed customers, not retrospective recomputation of global cohorts.

## Architecture

```text
UCI Excel workbook
  → Pandas validation + business rules
  → DuckDB SQL aggregation (order-level table)
  → RFM segment and cohort calculations
  → historical customer snapshots → temporal-embargo model evaluation
  → parquet artifacts + JSON metadata
  → five-page Streamlit / Plotly dashboard
```

Project files:
- `src/analytics.py`: cleaning, SQL order aggregation, RFM, monthly cohorts, labeled historical snapshots
- `src/model.py`: majority-class baseline, logistic regression, histogram gradient boosting, held-out evaluation
- `src/pipeline.py`: deterministic preparation of parquet data and model evaluation metadata
- `app.py`: read-only interactive dashboard
- `dashboard.py`: Streamlit Cloud entry point that downloads UCI data on initial startup if necessary
- `tests/`: analytics tests

## Local setup

Requires Python 3.11+.

```bash
python -m venv .venv
# activate .venv (Windows: .venv\\Scripts\\activate, macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

Download the original `Online Retail.xlsx` from the **UCI source link above** and place it in `data/Online Retail.xlsx`.

```bash
python -m src.pipeline
python -m pytest -q
streamlit run app.py
```

Alternatively, run `streamlit run dashboard.py` to let the app download the 22.6 MB UCI ZIP and process the source on first startup. Internet access is required for that initial download. The first run may take longer than later launches. Prepared parquet files and the raw workbook are intentionally **gitignored**.

## Dashboard pages

1. **Executive Overview** — customers, orders, accepted gross sales, repeat-purchase share and monthly trends
2. **Customer Segments** — sizes, contribution, RFM scatter and customer-level table
3. **Retention Analysis** — monthly cohort heatmap and month-specific retention curves
4. **Churn Risk** — historical test metrics, predicted inactivity probabilities and segment risk (only if a defensible evaluated model exists)
5. **Data Explorer** — filtered tables and CSV export

The Streamlit UI supports purchase date and RFM segment filters. Historical cohort calculations do not change when filtering purchases.

## Streamlit Community Cloud deployment

1. Open [share.streamlit.io](https://share.streamlit.io), sign in and select **Create app**.
2. Select GitHub repository `NandiniDhanrale/Customer-Segmentation-Retention-Analysis`, branch `main`.
3. Set **main file path** to `dashboard.py` (not `app.py`). Streamlit installs `requirements.txt` automatically.
4. Deploy and check logs. First startup downloads the public UCI workbook and prepares all artifacts in ephemeral storage. If UCI is inaccessible from the Cloud runtime, use local generation and an approved artifact delivery strategy instead.
5. Add your actual, verified live app URL and screenshots here only **after successful deployment**.

## Verification status

The project was written and pushed to GitHub, but the live dataset could not be downloaded and the full dependency stack was unavailable in the authoring environment. No full-data numerical results, model evaluation scores, public deployment URL or screenshot have been verified. Run the commands above and inspect the generated `data/prepared/metadata.json` and `data/prepared/model_scores.parquet` before citing results.

License of the underlying source is **CC BY 4.0** with attribution to Chen / UCI.
