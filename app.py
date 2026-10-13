import streamlit as st
import pandas as pd

st.set_page_config(page_title="Placement Package Predictor", page_icon="📈", layout="centered")

INTERCEPT = -7.368730395761633
COEF = {"CGPA": 1.53152658, "Projects": 0.2791608, "Internships": 0.49759255}

st.title("Placement Package Predictor")
st.caption("Multiple Linear Regression with scikit-learn")

with st.form("predict_form"):
    cgpa = st.number_input("CGPA (0 to 10)", min_value=0.0, max_value=10.0, value=None, step=0.1, placeholder="e.g. 8.7")
    projects = st.number_input("No. of Projects", min_value=0, value=None, step=1, placeholder="e.g. 4")
    internships = st.number_input("No. of Internships", min_value=0, value=None, step=1, placeholder="e.g. 2")
    submitted = st.form_submit_button("Predict Package", use_container_width=True)

if submitted:
    if cgpa is None or projects is None or internships is None:
        st.warning("Enter all three values")
    else:
        parts = {
            "Intercept": INTERCEPT,
            "CGPA": COEF["CGPA"] * cgpa,
            "Projects": COEF["Projects"] * projects,
            "Internships": COEF["Internships"] * internships,
        }
        total = sum(parts.values())
        st.metric("Predicted Placement Package", f"{max(total, 0):.2f} LPA")
        if total < 0:
            st.warning("The model gives a negative value here, which is outside its useful range.")
        st.subheader("How the prediction is built")
        st.bar_chart(pd.Series(parts, name="Contribution (LPA)"))
        st.caption("Package = Intercept + 1.5315 x CGPA + 0.2792 x Projects + 0.4976 x Internships")

st.divider()
st.subheader("Model performance")
c1, c2, c3 = st.columns(3)
c1.metric("R² Score", "0.9840")
c2.metric("MAE", "0.224")
c3.metric("MSE", "0.0882")
st.write("5-fold cross-validation R²: 0.973, 0.986, 0.985, 0.983, 0.976")
st.caption("Train/test split 80/20, random_state=2. CV scores are close to the test R², so the model is not overfitting.")

st.subheader("Feature correlation")
corr = pd.DataFrame(
    [[1.000, 0.919, 0.880], [0.919, 1.000, 0.732], [0.880, 0.732, 1.000]],
    index=["CGPA", "Projects", "Internships"],
    columns=["CGPA", "Projects", "Internships"],
)
st.dataframe(corr, use_container_width=True)
st.caption("Features are strongly correlated (multicollinearity), so individual coefficients should not be read as isolated effects.")

st.subheader("Limitations")
st.caption("Trained on a small dataset (about 60 rows), so predictions are estimates for learning purposes, not real placement guarantees. Values far outside the training data are unreliable.")
