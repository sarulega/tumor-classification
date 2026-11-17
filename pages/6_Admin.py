import os
import streamlit as st
from datetime import datetime
from common import setup, q, run, save_prediction, label, DB

X, y, Xte, yte, models, model, mname = setup("Admin", "⚙️")
st.title("⚙️ Admin dashboard")
t1, t2, t3 = st.tabs(["Users", "System information", "Data tools"])
with t1:
    st.dataframe(q("SELECT * FROM users"))
    with st.form("u", clear_on_submit=True):
        c = st.columns(2)
        u, r = c[0].text_input("Username"), c[1].selectbox("Role", ["Doctor", "Nurse", "Admin"])
        if st.form_submit_button("Add user") and u.strip():
            run("INSERT OR REPLACE INTO users VALUES(?,?)", (u.strip(), r)); st.rerun()
    users = q("SELECT username FROM users").username
    d = st.selectbox("Remove user", users)
    if st.button("Remove") and d != "admin":
        run("DELETE FROM users WHERE username=?", (d,)); st.rerun()
with t2:
    c = st.columns(4)
    c[0].metric("Patients", len(q("SELECT pid FROM patients"))); c[1].metric("Predictions", len(q("SELECT id FROM predictions")))
    c[2].metric("Users", len(q("SELECT username FROM users")))
    c[3].metric("DB size", f"{os.path.getsize(DB) / 1024:.0f} KB" if os.path.exists(DB) else "0 KB")
    st.write("**Models:** " + ", ".join(models))
    st.write("**Dataset:** Wisconsin Diagnostic Breast Cancer, 569 patients, 30 features")
with t3:
    if st.button("Load demo data (15 patients)"):
        for i, (_, row) in enumerate(Xte.head(15).iterrows()):
            pid = f"DEMO-{i + 1:02d}"
            run("INSERT OR IGNORE INTO patients VALUES(?,?,?,?,?,?)",
                (pid, f"Demo Patient {i + 1}", 30 + i * 3, "Female", "-", datetime.now().strftime("%Y-%m-%d %H:%M")))
            p = float(model.predict_proba(row.to_frame().T)[0, 1])
            save_prediction(pid, mname, p, label(p), row.to_dict())
        st.success("Demo data loaded. Open the Dashboard.")
    ok = st.checkbox("I understand this deletes all patients and predictions")
    if st.button("Clear all data") and ok:
        run("DELETE FROM predictions"); run("DELETE FROM patients"); st.success("All data cleared.")
