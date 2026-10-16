"""Streamlit Cloud entry point: prepare real UCI data once on first launch."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import streamlit as st

root = Path(__file__).parent
prepared = root/'data/prepared/metadata.json'
source = root/'data/Online Retail.xlsx'
uci_zip = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
if not prepared.exists():
    try:
        source.parent.mkdir(parents=True,exist_ok=True)
        if not source.exists():
            zip_path = root/'data/online-retail.zip'
            urlretrieve(uci_zip,zip_path)
            with ZipFile(zip_path) as archive:
                if 'Online Retail.xlsx' not in archive.namelist():
                    raise ValueError('Unexpected UCI archive contents')
                with archive.open('Online Retail.xlsx') as inp, source.open('wb') as out:
                    import shutil
                    shutil.copyfileobj(inp,out)
            zip_path.unlink(missing_ok=True)
        from src.pipeline import build
        build(source=source,out=prepared.parent)
    except Exception as error:
        st.error(f'Data preparation failed: {error}')
        st.markdown('For manual setup, download the UCI workbook, then run `python -m src.pipeline`.')
        st.stop()
# Importing app renders the dashboard from prepared artifacts.
import app
