"""Placement Probability Predictor — run with:  streamlit run app.py
Needs placement_models.joblib (created by the last cells of the notebook) in the same folder."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load("placement_models.joblib")


bundle = load_models()
rf, lr, meta = bundle["rf"], bundle["lr"], bundle["meta"]
rng = meta["ranges"]

st.title("🎓 Student Placement Probability")
st.caption("A decision-support tool for skill development — **not** a tool for selecting or rejecting students.")

# ---------------------------------------------------------------- inputs
with st.form("student_form"):
    st.subheader("Student details")
    c1, c2, c3 = st.columns(3)
    with c1:
        cgpa = st.slider("CGPA", rng["CGPA"][0], 10.0, 7.2, 0.01)
        apt = st.slider("Aptitude test score", int(rng["Aptitude_Test_Score"][0]), 100, 70)
        backlogs = st.slider("Backlogs", 0, int(rng["Backlogs"][1]), 0)
    with c2:
        coding = st.slider("Coding skills (1-10)", 1, 10, 6)
        comm = st.slider("Communication skills (1-10)", 1, 10, 6)
        soft = st.slider("Soft skills rating (1-10)", 1, 10, 5)
    with c3:
        intern = st.slider("Internships", int(rng["Internships"][0]), int(rng["Internships"][1]), 1)
        projects = st.slider("Projects", int(rng["Projects"][0]), int(rng["Projects"][1]), 4)
        certs = st.slider("Certifications", int(rng["Certifications"][0]), int(rng["Certifications"][1]), 2)
    c4, c5 = st.columns(2)
    degree = c4.selectbox("Degree", meta["categories"]["Degree"])
    branch = c5.selectbox("Branch", meta["categories"]["Branch"])
    submitted = st.form_submit_button("Predict placement probability", type="primary")

student = pd.DataFrame([{
    "CGPA": cgpa, "Internships": intern, "Projects": projects, "Coding_Skills": coding,
    "Communication_Skills": comm, "Aptitude_Test_Score": apt, "Soft_Skills_Rating": soft,
    "Certifications": certs, "Backlogs": backlogs, "Degree": degree, "Branch": branch}])

if not submitted:
    st.info("Fill in the details above and press **Predict**.")
    st.stop()

# ---------------------------------------------------------------- prediction
p_rf = float(rf.predict_proba(student)[0, 1])
p_lr = float(lr.predict_proba(student)[0, 1])
label = "Likely to be placed" if p_rf >= 0.5 else "Not likely to be placed (needs support)"

st.divider()
left, right = st.columns([1, 1])
with left:
    st.subheader("Prediction")
    (st.success if p_rf >= 0.5 else st.warning)(f"**{label}**")
    st.metric("Estimated placement probability (Random Forest)", f"{p_rf:.1%}")
    st.progress(min(max(p_rf, 0.0), 1.0))
    st.metric("Second opinion (Logistic Regression)", f"{p_lr:.1%}")
    st.caption(
        "Probability = the model's estimated *likelihood* based on patterns in past data, not a guarantee. "
        "The Random Forest is very confident on this dataset because the data follows strict cut-offs; "
        "the Logistic Regression gives a smoother estimate.")

with right:
    st.subheader("Readiness checklist")
    st.caption("The cut-offs below are patterns observed in this dataset (below them, almost no student was placed) — not official company rules.")
    checks = [
        ("CGPA ≥ 6.5", cgpa >= 6.5),
        ("Communication skills ≥ 5", comm >= 5),
        ("Coding skills ≥ 5", coding >= 5),
        ("Backlogs ≤ 1", backlogs <= 1),
        ("Aptitude score above 53", apt > 53),
        ("At least 4 projects", projects >= 4),
        ("At least 2 certifications", certs >= 2),
    ]
    for text, ok in checks:
        st.write(("✅ " if ok else "❌ ") + text)

# ---------------------------------------------------------------- what-if
st.subheader("Where would effort help most?")
rows = []
for feat, step in meta["steps"].items():
    s2 = student.copy()
    lo, hi = rng[feat]
    s2[feat] = float(np.clip(s2[feat].iloc[0] + step, lo, max(hi, s2[feat].iloc[0])))
    rows.append({"Improvement": f"{feat.replace('_', ' ')} {'+' if step > 0 else ''}{step}",
                 "New probability (LR)": lr.predict_proba(s2)[0, 1],
                 "Change": lr.predict_proba(s2)[0, 1] - p_lr})
plan = pd.DataFrame(rows).sort_values("Change", ascending=False)
st.dataframe(plan.style.format({"New probability (LR)": "{:.1%}", "Change": "{:+.1%}"}),
             hide_index=True)
st.caption("These are model associations, not guaranteed outcomes. Use them as suggested focus areas "
           "and discuss with a placement counsellor.")
