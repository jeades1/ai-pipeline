# ui/app.py
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Biomarker Discovery Viewer", layout="wide")

summary_p = Path("artifacts/run_summary.json")
if not summary_p.exists():
    st.error(
        "No artifacts/run_summary.json found. Generate artifacts with the pipeline and refresh."
    )
    st.stop()

with summary_p.open() as f:
    meta = json.load(f)

st.title("Evidence Graph — Run Summary")
st.caption(
    f"Context: {meta.get('context')} · Used real data: {meta.get('used_real_data')}"
)

cols = st.columns(4)
cols[0].metric("Assoc records", f"{meta.get('n_assoc_records', 0):,}")
cols[1].metric("Ranked biomarkers", f"{meta.get('n_ranked', 0):,}")
cols[2].metric("Experiments", f"{meta.get('n_experiments', 0):,}")
ds = meta.get("data_sources", {})
cols[3].write("**Data sources**")
cols[3].write(
    f"- GEO DEGs: {ds.get('geo_deg')}\n"
    f"- MIMIC labels: {ds.get('mimic')}\n"
    f"- Reactome: {ds.get('reactome')}\n"
    f"- CellPhoneDB: {ds.get('cellphonedb')}"
)

st.divider()

tab1, tab2, tab3 = st.tabs(["Biomarkers", "Enrichment", "Experiments"])

with tab1:
    # read top 50 from biomarker_cards by scanning markdown names
    cards_dir = Path(meta["artifacts"]["biomarker_cards"])
    items = sorted(cards_dir.glob("*.md"))
    st.write(f"Found {len(items)} biomarker cards. Showing first 20:")
    for p in items[:20]:
        st.markdown(p.read_text())

with tab2:
    st.write("Pathway enrichment (placeholder; replace mapping when available):")
    enr = pd.DataFrame(meta.get("enrichment_top", []))
    if not enr.empty:
        st.dataframe(enr)
    else:
        st.info("No enrichment computed.")

with tab3:
    st.write("Top experiment cards:")
    e_dir = Path(meta["artifacts"]["experiment_cards"])
    e_items = sorted(e_dir.glob("*.md"))
    for p in e_items:
        st.markdown(p.read_text())

st.divider()
st.subheader("Benchmark comparison")
cmp_p = Path("artifacts/comparison_report.md")
if cmp_p.exists():
    st.markdown(cmp_p.read_text())
else:
    st.info("comparison_report.md not found.")
