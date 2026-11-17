"""Home dashboard. Run with:  streamlit run app.py"""
import html
import streamlit as st
import matplotlib.pyplot as plt
from common import setup, q, header, kpi, badge, COLORS

X, y, Xte, yte, models, model, name = setup("Dashboard", "🏠")
header("Tumor Diagnostic Support", "Classify tumors as benign or malignant with probabilistic machine learning")

pat = q("SELECT * FROM patients")
pr = q("SELECT id,pid,time,model,p_mal,decision FROM predictions ORDER BY id DESC")
high = int(pr.decision.str.startswith("HIGH").sum()) if len(pr) else 0
unc = int(pr.decision.str.startswith("UNCERTAIN").sum()) if len(pr) else 0

c = st.columns(4)
kpi(c[0], "Total patients", len(pat), "registered", "#0F766E")
kpi(c[1], "Predictions made", len(pr), "all models", "#2563EB")
kpi(c[2], "High-risk cases", high, f"{unc} uncertain", COLORS["HIGH RISK"])
kpi(c[3], "Average risk score", f"{pr.p_mal.mean():.0%}" if len(pr) else "n/a", "mean P(malignant)", "#b7791f")
st.write("")

if pr.empty:
    st.info("No records yet. Register a patient under Patients and run a prediction, "
            "or load demo data from the Admin page.")
else:
    a, b = st.columns(2)
    counts = pr.decision.str.split(":").str[0].str.strip().value_counts()
    with a:
        st.subheader("Risk distribution")
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        fig.patch.set_alpha(0)
        ax.pie(counts, labels=counts.index, colors=[COLORS.get(k, "#5B7479") for k in counts.index],
               autopct="%d%%", textprops={"color": "#12303A", "fontsize": 9},
               wedgeprops=dict(width=.42, edgecolor="white"))
        ax.text(0, 0, str(len(pr)), ha="center", va="center", fontsize=20, weight="bold", color="#12303A")
        st.pyplot(fig)
    with b:
        st.subheader("Predictions per day")
        st.bar_chart(pr.assign(day=pr.time.str[:10]).groupby("day").size(), color="#0F766E")

    st.subheader("Recent predictions")
    rows = "".join(
        f"<tr><td>{html.escape(str(r.pid))}</td><td>{html.escape(str(r.time))}</td>"
        f"<td>{html.escape(str(r.model))}</td><td>{r.p_mal * 100:.1f}%</td><td>{badge(r.decision)}</td></tr>"
        for r in pr.head(8).itertuples())
    st.markdown('<div class="card"><table class="rec"><tr><th>Patient</th><th>Time</th><th>Model</th>'
                f'<th>P(malignant)</th><th>Result</th></tr>{rows}</table></div>', unsafe_allow_html=True)
