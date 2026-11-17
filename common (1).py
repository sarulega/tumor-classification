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


# ---------------------------------------------------------------- UI helpers
import html as _html

COLORS = {"LOW RISK": "#1d7a72", "UNCERTAIN": "#b7791f", "HIGH RISK": "#8a2f6b"}

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family:'Manrope',sans-serif;}
.stApp {background:#F3F7F7;}
.block-container {padding-top:1.4rem; max-width:1250px;}
h1, h2, h3 {color:#12303A; letter-spacing:-.01em;}
[data-testid=stSidebar] {background:#12303A;}
[data-testid=stSidebar] * {color:#E6F1F0;}
[data-testid=stSidebar] [data-baseweb=select] * {color:#12303A;}
[data-testid=stSidebar] [data-testid=stSidebarNav] a {border-radius:10px; padding:.35rem .6rem;}
[data-testid=stSidebar] [data-testid=stSidebarNav] a:hover {background:rgba(255,255,255,.10);}
[data-testid=stSidebar] [aria-current=page] {background:#0F766E !important;}
[data-testid=stSidebar] .stAlert {background:rgba(255,255,255,.08); border:0;}
div[data-testid=stMetric] {background:#fff; border:1px solid #DCE7E7; border-radius:14px; padding:14px 18px;}
.hero {background:linear-gradient(120deg,#0F766E 0%,#12303A 100%); color:#fff; border-radius:18px;
  padding:26px 30px; margin-bottom:20px;}
.hero h1 {color:#fff; margin:0; font-size:1.9rem; font-weight:800;}
.hero p {color:#CFE8E5; margin:.35rem 0 0 0; font-size:1rem;}
.kpi {background:#fff; border:1px solid #DCE7E7; border-left:6px solid var(--c); border-radius:14px;
  padding:16px 18px;}
.kpi .l {color:#5B7479; font-size:.85rem; font-weight:600;}
.kpi .v {color:#12303A; font-size:2rem; font-weight:800; line-height:1.15;}
.kpi .n {color:#5B7479; font-size:.8rem;}
.card {background:#fff; border:1px solid #DCE7E7; border-radius:14px; padding:18px 20px;}
.badge {display:inline-block; padding:3px 11px; border-radius:999px; font-size:.78rem; font-weight:700;
  color:#fff;}
table.rec {width:100%; border-collapse:collapse; font-size:.9rem;}
table.rec th {text-align:left; color:#5B7479; font-weight:600; padding:8px 10px; border-bottom:2px solid #DCE7E7;}
table.rec td {padding:9px 10px; border-bottom:1px solid #EAF1F1; color:#12303A;}
.stButton > button, .stDownloadButton > button {background:#0F766E; color:#fff; border:0; border-radius:10px;
  font-weight:700; padding:.5rem 1.2rem;}
.stButton > button:hover, .stDownloadButton > button:hover {background:#12303A; color:#fff;}
</style>"""


def header(title, subtitle=""):
    """Gradient banner shown at the top of a page."""
    st.markdown(f'<div class="hero"><h1>{_html.escape(title)}</h1><p>{_html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def kpi(col, label, value, note="", color="#0F766E"):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{_html.escape(label)}</div>'
                 f'<div class="v">{_html.escape(str(value))}</div><div class="n">{_html.escape(note)}</div></div>',
                 unsafe_allow_html=True)


def badge(decision):
    k = decision.split(":")[0].strip()
    return f'<span class="badge" style="background:{COLORS.get(k, "#5B7479")}">{_html.escape(k.title())}</span>'


def result_card(p, decision):
    """Large result card for a single prediction."""
    k = decision.split(":")[0].strip()
    c = COLORS.get(k, "#5B7479")
    st.markdown(
        f'<div class="card" style="border-left:8px solid {c}">'
        f'<div style="color:#5B7479;font-weight:600">Probability of malignancy</div>'
        f'<div style="font-size:2.6rem;font-weight:800;color:{c};line-height:1.1">{p:.1%}</div>'
        f'<div style="margin:6px 0 10px 0">{badge(decision)}</div>'
        f'<div style="background:#E3EDED;border-radius:999px;height:12px;overflow:hidden">'
        f'<div style="width:{p * 100:.1f}%;background:{c};height:12px"></div></div>'
        f'<div style="margin-top:10px;color:#12303A">{_html.escape(decision.split(":", 1)[-1].strip())}</div></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------- page setup
def setup(title, icon="🩺"):
    st.set_page_config(page_title=f"{title} | Tumor Diagnostic Support", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    X, y, Xte, yte, models = load_everything()
    st.sidebar.markdown("### 🩺 Tumor Diagnostic Support")
    name = st.sidebar.selectbox("Model", list(models), key="model_choice")
    st.sidebar.caption(f"Uncertain band: {LOW:.0%} to {HIGH:.0%}")
    st.sidebar.warning("Educational project. Not for clinical use.")
    return X, y, Xte, yte, models, models[name], name
