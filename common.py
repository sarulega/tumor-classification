"""Shared models, database, PDF report and page setup."""
import io, json, sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOW, HIGH, DB = 0.10, 0.90, "clinic.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS patients(pid TEXT PRIMARY KEY,name TEXT,age INT,gender TEXT,phone TEXT,created TEXT);
CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,pid TEXT,time TEXT,model TEXT,
  p_mal REAL,decision TEXT,features TEXT,img1 BLOB,img2 BLOB);
CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY,role TEXT);
INSERT OR IGNORE INTO users VALUES('admin','Admin'),('doctor','Doctor');"""


@st.cache_resource
def load_everything():
    d = load_breast_cancer(as_frame=True)
    X, y = d.data, 1 - d.target  # 1 = malignant
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models = {"Gaussian Naive Bayes": GaussianNB(),
              "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5))}
    for m in models.values():
        m.fit(Xtr, ytr)
    return X, y, Xte, yte, models


def decide(p):
    if p >= HIGH:
        return "HIGH RISK: likely malignant. Urgent specialist follow-up advised.", "error"
    if p <= LOW:
        return "LOW RISK: likely benign. Routine follow-up advised.", "success"
    return "UNCERTAIN: refer to a specialist for further tests.", "warning"


def label(p):
    return decide(p)[0].split(":")[0]


def nb_evidence(row, models):
    nb = models["Gaussian Naive Bayes"]
    mu, var, x = nb.theta_, nb.var_, row.values.astype(float)
    ll = lambda k: -0.5 * np.log(2 * np.pi * var[k]) - (x - mu[k]) ** 2 / (2 * var[k])
    return pd.Series(ll(1) - ll(0), index=row.index).sort_values(key=abs, ascending=False)


# ---------------------------------------------------------------- database
def _conn():
    c = sqlite3.connect(DB)
    c.executescript(SCHEMA)
    return c


def q(sql, args=()):
    c = _conn()
    df = pd.read_sql(sql, c, params=args)
    c.close()
    return df


def run(sql, args=()):
    c = _conn()
    c.execute(sql, args)
    c.commit()
    c.close()


def next_pid():
    ids, n = set(q("SELECT pid FROM patients").pid), 1
    while f"P-{n:04d}" in ids:
        n += 1
    return f"P-{n:04d}"


def save_prediction(pid, model, p, decision, feats, img1=None, img2=None):
    run("INSERT INTO predictions(pid,time,model,p_mal,decision,features,img1,img2) VALUES(?,?,?,?,?,?,?,?)",
        (pid, datetime.now().strftime("%Y-%m-%d %H:%M"), model, float(p), decision,
         json.dumps(feats, default=float), img1, img2))


def blob(v):
    return v if isinstance(v, bytes) else None


# ---------------------------------------------------------------- PDF report
def report_pdf(pat, rec, models):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    p = float(rec["p_mal"])
    ev = nb_evidence(pd.Series(json.loads(rec["features"])), models).head(6)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.08, .95, "Tumor Diagnostic Report", fontsize=20, weight="bold")
        fig.text(.08, .925, f"Generated {datetime.now():%Y-%m-%d %H:%M}", fontsize=9, color="gray")
        lines = [f"Patient: {pat['name']} ({pat['pid']})", f"Age / Gender: {pat['age']} / {pat['gender']}",
                 f"Model: {rec['model']}    Date: {rec['time']}",
                 f"P(malignant): {p:.1%}    P(benign): {1 - p:.1%}", f"Decision: {decide(p)[0]}"]
        for i, t in enumerate(lines):
            fig.text(.08, .88 - i * .027, t, fontsize=11, weight="bold" if i == 4 else None)
        ax = fig.add_axes([.08, .70, .84, .03])
        ax.barh([0], [1 - p], color="#1d7a72"); ax.barh([0], [p], left=[1 - p], color="#8a2f6b")
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_title("benign (green)  vs  malignant (purple)", fontsize=8)
        ax = fig.add_axes([.34, .46, .58, .17])
        ax.barh(ev.index[::-1], ev.values[::-1], color=["#8a2f6b" if v > 0 else "#1d7a72" for v in ev.values[::-1]])
        ax.axvline(0, color="k", lw=.8); ax.set_title("Top features driving the decision", fontsize=9)
        ax.tick_params(labelsize=7)
        for k, (key, x0) in enumerate([("img1", .08), ("img2", .52)]):
            b = blob(rec.get(key))
            if b:
                a = fig.add_axes([x0, .16, .40, .24]); a.imshow(Image.open(io.BytesIO(b))); a.axis("off")
                a.set_title(f"Scan image {k + 1}", fontsize=8)
        fig.text(.08, .07, "Educational project. Not for real clinical use or diagnosis.", fontsize=8, color="red")
        pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- page setup
def setup(title, icon="🩺"):
    st.set_page_config(page_title=f"{title} | Tumor Diagnostic Support", page_icon=icon, layout="wide")
    st.markdown("""<style>
    [data-testid=stSidebar]{background:rgba(31,107,255,.08)}
    div[data-testid=stMetric]{background:rgba(31,107,255,.07);border:1px solid rgba(31,107,255,.25);
      border-radius:12px;padding:14px 18px}
    </style>""", unsafe_allow_html=True)
    X, y, Xte, yte, models = load_everything()
    st.sidebar.markdown("### 🩺 Tumor Diagnostic Support")
    name = st.sidebar.selectbox("Model", list(models), key="model_choice")
    st.sidebar.caption(f"Uncertain band: {LOW:.0%} to {HIGH:.0%}")
    st.sidebar.warning("Educational project. Not for clinical use.")
    return X, y, Xte, yte, models, models[name], name
