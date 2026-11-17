"""Home dashboard. Run with:  streamlit run app.py"""
import streamlit as st
import matplotlib.pyplot as plt
from common import setup, q

X, y, Xte, yte, models, model, name = setup("Dashboard", "🏠")
st.title("🏠 Dashboard")
st.caption("System overview and statistics")
pat = q("SELECT * FROM patients")
pr = q("SELECT id,pid,time,model,p_mal,decision FROM predictions ORDER BY id DESC")
c = st.columns(4)
c[0].metric("Total patients", len(pat))
c[1].metric("Predictions made", len(pr))
c[2].metric("High-risk cases", int((pr.decision == "HIGH RISK").sum()))
c[3].metric("Avg risk score", f"{pr.p_mal.mean():.0%}" if len(pr) else "n/a")
if pr.empty:
    st.info("No records yet. Register a patient and run a prediction, or load demo data from the Admin page.")
else:
    a, b = st.columns(2)
    counts = pr.decision.value_counts()
    colors = {"LOW RISK": "#1d7a72", "UNCERTAIN": "#b7791f", "HIGH RISK": "#8a2f6b"}
    with a:
        st.subheader("Risk distribution")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.pie(counts, labels=counts.index, colors=[colors[k] for k in counts.index], autopct="%d%%",
               wedgeprops=dict(width=.45))
        ax.text(0, 0, str(len(pr)), ha="center", va="center", fontsize=18, weight="bold")
        st.pyplot(fig)
    with b:
        st.subheader("Predictions per day")
        st.bar_chart(pr.assign(day=pr.time.str[:10]).groupby("day").size())
    st.subheader("Recent predictions")
    st.dataframe(pr.head(8).assign(p_mal=lambda d: (d.p_mal * 100).round(1)).rename(columns={"p_mal": "P(malignant) %"}))
