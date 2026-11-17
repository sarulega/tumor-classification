import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from common import setup, q, decide, label, nb_evidence, save_prediction, report_pdf

X, y, Xte, yte, models, model, mname = setup("New Prediction", "🧪")
FEATS = list(X.columns)
st.title("🧪 New prediction")
pats = q("SELECT pid,name FROM patients")
if pats.empty:
    st.warning("Register a patient first on the Patients page.")
    st.stop()
names = dict(zip(pats.pid, pats.name))
pid = st.selectbox("Patient", pats.pid, format_func=lambda i: f"{i} · {names[i]}")


def load_sample(kind):
    idx = yte[yte == kind].index[0]
    for f in FEATS:
        st.session_state[f] = float(Xte.loc[idx, f])


c1, c2 = st.columns(2)
c1.button("Load sample benign case", on_click=load_sample, args=(0,))
c2.button("Load sample malignant case", on_click=load_sample, args=(1,))
vals = {}
for title, fs in {"Mean values": FEATS[:10], "Error values": FEATS[10:20], "Worst values": FEATS[20:]}.items():
    with st.expander(f"Tumor measurements: {title}", expanded=(title == "Worst values")):
        cols = st.columns(2)
        for i, f in enumerate(fs):
            vals[f] = cols[i % 2].number_input(f, 0.0, float(X[f].max()) * 1.5,
                                               st.session_state.get(f, float(X[f].median())), key=f, format="%.4f")

st.subheader("Scan images (optional)")
u1, u2 = st.columns(2)
f1 = u1.file_uploader("Scan image 1", type=["png", "jpg", "jpeg"], key="im1")
f2 = u2.file_uploader("Scan image 2", type=["png", "jpg", "jpeg"], key="im2")
for col, f in ((u1, f1), (u2, f2)):
    if f:
        col.image(f)
st.caption("Images are stored with the record and printed in the report. The model classifies from the "
           "measurements above, not from the image pixels.")

if st.button("Classify", type="primary"):
    row = pd.Series(vals)[FEATS]
    p = float(model.predict_proba(row.to_frame().T)[0, 1])
    b1, b2 = (f1.getvalue() if f1 else None), (f2.getvalue() if f2 else None)
    save_prediction(pid, mname, p, label(p), row.to_dict(), b1, b2)
    st.session_state.last = dict(pid=pid, time=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), model=mname, p_mal=p,
                                 features=json.dumps(row.to_dict()), img1=b1, img2=b2)

if "last" in st.session_state:
    r = st.session_state.last
    p = r["p_mal"]
    msg, kind = decide(p)
    st.divider()
    st.subheader(f"Result for {r['pid']}")
    a, b = st.columns([1, 2])
    a.metric("P(malignant)", f"{p:.1%}")
    a.metric("P(benign)", f"{1 - p:.1%}")
    getattr(b, kind)(msg)
    b.dataframe(pd.DataFrame({"Class": ["Benign", "Malignant"], "Probability": [(1 - p) * 100, p * 100]}),
                column_config={"Probability": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})
    ev = nb_evidence(pd.Series(json.loads(r["features"])), models).head(6)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(ev.index[::-1], ev.values[::-1], color=["#8a2f6b" if v > 0 else "#1d7a72" for v in ev.values[::-1]])
    ax.axvline(0, color="k", lw=.8); ax.set_xlabel("evidence: left = benign, right = malignant")
    st.pyplot(fig)
    pat = q("SELECT * FROM patients WHERE pid=?", (r["pid"],)).iloc[0].to_dict()
    st.download_button("Download PDF report", report_pdf(pat, r, models), f"report_{r['pid']}.pdf", "application/pdf")
    st.success("Saved to Prediction History.")
