import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from common import setup

X, y, Xte, yte, models, model, mname = setup("Model Comparison", "📊")
st.title("📊 Model comparison")
st.caption("Naive Bayes vs other models on a held-out 25% test set (143 patients).")
rows, probs = [], {}
for n, m in models.items():
    pr = m.predict_proba(Xte)[:, 1]
    pred = (pr >= .5).astype(int)
    probs[n] = pr
    rows.append({"Model": n, "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
                 "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred), "ROC-AUC": roc_auc_score(yte, pr)})
df = pd.DataFrame(rows).set_index("Model").round(4)
st.dataframe(df)
st.bar_chart(df[["Accuracy", "Recall", "F1"]])
c1, c2 = st.columns(2)
cm = confusion_matrix(yte, (probs[mname] >= .5).astype(int))
fig, ax = plt.subplots(figsize=(4, 3.4)); ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, v, ha="center", va="center")
ax.set_xticks([0, 1], ["Benign", "Malignant"]); ax.set_yticks([0, 1], ["Benign", "Malignant"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix: {mname}")
c1.pyplot(fig)
fig, ax = plt.subplots(figsize=(4, 3.4))
for n, pr in probs.items():
    f_, t_, _ = roc_curve(yte, pr); ax.plot(f_, t_, label=n)
ax.plot([0, 1], [0, 1], "k--"); ax.legend(fontsize=7); ax.set_title("ROC curves")
c2.pyplot(fig)
st.subheader("Threshold trade-off")
t = st.slider("Decision threshold for malignant", .05, .95, .5, .05)
tn, fp, fn, tp = confusion_matrix(yte, (probs[mname] >= t).astype(int)).ravel()
a = st.columns(3)
a[0].metric("Missed cancers (FN)", int(fn)); a[1].metric("False alarms (FP)", int(fp)); a[2].metric("Recall", f"{tp / (tp + fn):.1%}")
