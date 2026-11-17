import streamlit as st
from common import setup, q, run, blob, report_pdf, decide

X, y, Xte, yte, models, model, mname = setup("History and Reports", "🗂️")
st.title("🗂️ Prediction history and reports")
h = q("SELECT id,pid,time,model,p_mal,decision FROM predictions ORDER BY id DESC")
if h.empty:
    st.info("No predictions yet.")
    st.stop()
f = st.selectbox("Filter by patient", ["All"] + sorted(h.pid.unique()))
if f != "All":
    h = h[h.pid == f]
st.dataframe(h.assign(p_mal=(h.p_mal * 100).round(1)).rename(columns={"p_mal": "P(malignant) %"}))
st.download_button("Download history (CSV)", h.to_csv(index=False), "prediction_history.csv")
st.divider()
sel = st.selectbox("Open a prediction (ID)", h.id)
rec = q("SELECT * FROM predictions WHERE id=?", (int(sel),)).iloc[0].to_dict()
pat = q("SELECT * FROM patients WHERE pid=?", (rec["pid"],))
c = st.columns(3)
c[0].metric("Patient", rec["pid"]); c[1].metric("P(malignant)", f"{rec['p_mal']:.1%}"); c[2].metric("Decision", rec["decision"])
imgs = [blob(rec["img1"]), blob(rec["img2"])]
if any(imgs):
    cols = st.columns(2)
    for col, b in zip(cols, imgs):
        if b:
            col.image(b)
if not pat.empty:
    st.download_button("Download PDF report", report_pdf(pat.iloc[0].to_dict(), rec, models),
                       f"report_{rec['pid']}_{sel}.pdf", "application/pdf")
if st.button("Delete this prediction"):
    run("DELETE FROM predictions WHERE id=?", (int(sel),))
    st.rerun()
