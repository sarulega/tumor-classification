import streamlit as st
from datetime import datetime
from common import setup, q, run, next_pid

setup("Patients", "👥")
st.title("👥 Patients")
t1, t2 = st.tabs(["Register patient", "Manage patients"])
with t1:
    with st.form("reg", clear_on_submit=True):
        st.text_input("Patient ID (auto)", next_pid(), disabled=True)
        name = st.text_input("Full name")
        c = st.columns(3)
        age = c[0].number_input("Age", 1, 120, 40)
        gender = c[1].selectbox("Gender", ["Female", "Male", "Other"])
        phone = c[2].text_input("Phone")
        if st.form_submit_button("Register", type="primary"):
            if not name.strip():
                st.error("Name is required.")
            else:
                pid = next_pid()
                run("INSERT INTO patients VALUES(?,?,?,?,?,?)",
                    (pid, name.strip(), int(age), gender, phone, datetime.now().strftime("%Y-%m-%d %H:%M")))
                st.success(f"Registered {name} with ID {pid}.")
with t2:
    s = st.text_input("Search by name or ID")
    df = q("SELECT * FROM patients ORDER BY created DESC")
    if s:
        df = df[df.apply(lambda r: s.lower() in f"{r['pid']} {r['name']}".lower(), axis=1)]
    st.dataframe(df)
    if not df.empty:
        d = st.selectbox("Delete patient", df.pid)
        if st.button("Delete selected patient and their predictions"):
            run("DELETE FROM predictions WHERE pid=?", (d,))
            run("DELETE FROM patients WHERE pid=?", (d,))
            st.rerun()
