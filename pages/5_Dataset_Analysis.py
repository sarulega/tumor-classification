import streamlit as st
import matplotlib.pyplot as plt
from common import setup

X, y, Xte, yte, models, model, mname = setup("Dataset Analysis", "🔬")
st.title("🔬 Dataset analysis and visualisation")
t1, t2, t3 = st.tabs(["Overview", "Feature distributions", "Correlations"])
with t1:
    c = st.columns(3)
    c[0].metric("Patients", len(X)); c[1].metric("Features", X.shape[1]); c[2].metric("Missing values", int(X.isna().sum().sum()))
    st.bar_chart(y.map({0: "Benign", 1: "Malignant"}).value_counts())
    st.dataframe(X.describe().T.round(3))
    st.download_button("Download dataset (CSV)", X.assign(malignant=y).to_csv(index=False), "dataset.csv")
with t2:
    f = st.selectbox("Feature", list(X.columns), index=20)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k, col, nm in [(0, "#1d7a72", "Benign"), (1, "#8a2f6b", "Malignant")]:
        ax.hist(X[f][y == k], bins=25, alpha=.55, color=col, label=nm)
    ax.set_xlabel(f); ax.legend()
    st.pyplot(fig)
    st.caption("Separated curves mean the feature is useful for Naive Bayes.")
with t3:
    cols = list(X.columns[:10])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(X[cols].corr(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(10), cols, rotation=90, fontsize=7); ax.set_yticks(range(10), cols, fontsize=7)
    fig.colorbar(im); st.pyplot(fig)
    st.caption("Strong correlations (e.g. radius, perimeter, area) break Naive Bayes' independence assumption.")
