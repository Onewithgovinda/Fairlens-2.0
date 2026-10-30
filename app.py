import json
import os
from pathlib import Path

from dotenv import load_dotenv

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ai_explainer import generate_loan_explanation
load_dotenv()
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "approval_model_tuned.joblib"
METRICS = ROOT / "approval_metrics.json"
DATA = ROOT / "data" / "loan_approval_dataset.csv"

st.set_page_config(
    page_title="Fairlens | Explainable Loan Approval",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0B1120;
        color: #E5E7EB;
    }

    [data-testid="stHeader"] {
        background: #0B1120;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263244;
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #111827);
        color: white;
        padding: 1.6rem;
        border: 1px solid #334155;
        border-radius: 18px;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: #FFFFFF;
        margin: 0;
    }

    .hero p {
        color: #CBD5E1;
        margin-top: .4rem;
    }

    h1, h2, h3, h4 {
        color: #F8FAFC;
    }

    p, label, .stMarkdown {
        color: #CBD5E1;
    }

    [data-testid="stMetric"] {
        background: #172033;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #38BDF8;
    }

    [data-testid="stMetricLabel"] {
        color: #CBD5E1;
    }

    [data-testid="stForm"],
    [data-testid="stExpander"] {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 14px;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    .stTextArea textarea {
        background: #172033;
        color: #FFFFFF;
        border-color: #475569;
    }

    .stButton button,
    .stFormSubmitButton button {
        background: #2563EB;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover {
        background: #1D4ED8;
        color: white;
    }

    .privacy-note {
        background: #292211;
        color: #FDE68A;
        border: 1px solid #854D0E;
        padding: .8rem 1rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>Fairlens</h1><p>Explainable Loan Approval Research Prototype</p></div>',
    unsafe_allow_html=True,
)
st.warning(
    "Educational prototype only. This is not a bank decision or a reliable estimate for real lending."
)
# ---------------- DEMO LOGIN ----------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Fairlens Login")
    st.write("Sign in to access the loan assessment dashboard.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_clicked = st.form_submit_button("Login")

    if login_clicked:
        correct_username = os.getenv("FAIRLENS_DEMO_USERNAME")
        correct_password = os.getenv("FAIRLENS_DEMO_PASSWORD")

        if not correct_username or not correct_password:
            st.error("Login configuration is missing. Check your .env file.")
        elif username == correct_username and password == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect username or password.")

    st.stop()

if st.sidebar.button("Log out"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.success("Logged in: Demo User")

# ---------------- END DEMO LOGIN ----------------

page = st.sidebar.radio(
    "Navigate",
    ["Applicant Assessment", "Model Report", "Fairness Audit", "About & Limitations"],
)
st.sidebar.caption("DEV-P02 · Deviathon 2.0")


def assets():
    if not MODEL.exists() or not METRICS.exists():
        st.error("Model files missing. Follow README.md to train the model first.")
        st.stop()
    return joblib.load(MODEL), json.loads(METRICS.read_text(encoding="utf-8"))


def data():
    fairness_path = ROOT / "fairness_test_predictions.csv"

    if not fairness_path.exists():
        return None

    d = pd.read_csv(fairness_path)

    d.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in d.columns
    ]

    return d

def target_value(v):
    s = str(v).strip().lower()
    if s in {"1", "yes", "y", "approved", "approve", "true"}:
        return 1
    if s in {"0", "no", "n", "rejected", "reject", "false"}:
        return 0
    try:
        n = float(s)
        return int(n) if n in (0, 1) else np.nan
    except ValueError:
        return np.nan


def label(s):
    return s.replace("_", " ").title()


def safe_default(meta, feature, fallback):
    return meta.get("form_defaults", {}).get(feature, fallback)


if page == "Applicant Assessment":
    model, meta = assets()
    st.subheader("Applicant assessment")
    st.write(
        "Complete the applicant profile and upload a PAN-card image to unlock the demo assessment."
    )

    st.markdown("### 1. Applicant profile")
    with st.container(border=True):
        p1, p2, p3 = st.columns(3)
        applicant_name = p1.text_input("Applicant full name *", placeholder="Enter sample applicant name")
        age = p2.number_input("Age *", min_value=18, max_value=100, value=25, step=1)
        employment_years = p3.number_input(
            "Employment length (years) *", min_value=0.0, max_value=60.0, value=2.0, step=0.5
        )

    st.markdown("### 2. Income and requested loan")
    with st.container(border=True):
        a1, a2, a3 = st.columns(3)
        monthly_income = a1.number_input(
            "Monthly income (₹) *", min_value=0.0, value=50000.0, step=5000.0
        )
        coapplicant_income = a2.number_input(
            "Co-applicant monthly income (₹)", min_value=0.0, value=0.0, step=5000.0
        )
        loan_amount = a3.number_input(
            "Loan amount requested (₹) *", min_value=1.0, value=200000.0, step=10000.0
        )
        b1, b2, b3 = st.columns(3)
        loan_term = b1.number_input(
            "Loan term (years) *", min_value=1, max_value=40, value=12, step=1
        )
        home_ownership = b2.selectbox(
            "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )
        loan_purpose = b3.selectbox(
            "Loan purpose", ["EDUCATION", "HOME", "PERSONAL", "MEDICAL", "BUSINESS", "OTHER"]
        )

    st.markdown("### 3. Credit profile")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        loan_grade = c1.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])
        interest_rate = c2.number_input(
            "Interest rate (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.25
        )
        credit_history_years = c3.number_input(
            "Credit history length (years)", min_value=0.0, max_value=80.0, value=3.0, step=1.0
        )
        c4, c5 = st.columns(2)
        previous_default = c4.selectbox("Previous default on file?", ["No", "Yes"])
        existing_debt = c5.number_input(
            "Existing monthly debt payments (₹)", min_value=0.0, value=0.0, step=1000.0
        )

    st.markdown("### 4. Financial assets used by this model")
    st.caption(
        "These fields match the dataset used to train the approval model. Enter realistic sample values in rupees."
    )
    with st.container(border=True):
        d1, d2, d3 = st.columns(3)
        dependents = d1.number_input("Number of dependents", min_value=0, max_value=20, value=1, step=1)
        education = d2.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = d3.selectbox("Self employed", ["No", "Yes"])
        e1, e2, e3 = st.columns(3)
        residential_assets = e1.number_input(
            "Residential assets value (₹)", min_value=0.0, value=0.0, step=100000.0
        )
        commercial_assets = e2.number_input(
            "Commercial assets value (₹)", min_value=0.0, value=0.0, step=100000.0
        )
        luxury_assets = e3.number_input(
            "Luxury assets value (₹)", min_value=0.0, value=0.0, step=100000.0
        )
        e4, e5 = st.columns(2)
        bank_assets = e4.number_input(
            "Bank asset value (₹)", min_value=0.0, value=50000.0, step=50000.0
        )
        cibil_score = e5.number_input(
            "CIBIL score", min_value=300, max_value=900, value=750, step=1
        )

    st.markdown("### 5. Mandatory PAN-card upload")
    st.markdown(
        '<div class="privacy-note"><b>Privacy reminder:</b> For a public hackathon demo, use a synthetic/sample PAN image only. '
        "The upload is required to continue but this prototype does not verify identity or PAN validity. "
        "The image is processed in the current app session and is not written to a file by this code.</div>",
        unsafe_allow_html=True,
    )
    pan_file = st.file_uploader(
        "Upload PAN card image (PNG/JPG) *",
        type=["png", "jpg", "jpeg"],
        help="Required for the demo flow. Use a synthetic sample image, not a real person's identity document.",
    )
    if pan_file is not None:
        st.success(f"Image uploaded: {pan_file.name}")
        try:
            st.image(pan_file, caption="Uploaded sample image — not identity verified", width=320)
            pan_file.seek(0)
        except Exception:
            st.warning("The file was uploaded, but the image preview could not be displayed.")

    st.caption("* Required fields. The extra profile fields are collected for the demo form; the model only uses the features it was trained on.")

    submitted = st.button(
        "Verify upload & estimate approval",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        errors = []
        if not applicant_name.strip():
            errors.append("Enter the applicant's name.")
        if pan_file is None:
            errors.append("Upload a PAN-card image to continue.")
        if monthly_income <= 0:
            errors.append("Monthly income must be greater than ₹0.")
        if loan_amount <= 0:
            errors.append("Loan amount must be greater than ₹0.")
            if existing_debt > monthly_income:
                errors.append(
                "Existing monthly debt cannot exceed monthly income."
            )
        if errors:
            for error in errors:
                st.error(error)
        else:
            features = meta["feature_columns"]
            # Map form values only to columns the model was actually trained on.
            # Profile fields not represented in the training data are not model inputs.
            supplied = {
                "no_of_dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": monthly_income * 12,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets,
                "commercial_assets_value": commercial_assets,
                "luxury_assets_value": luxury_assets,
                "bank_asset_value": bank_assets,
            }
            defaults = meta.get("form_defaults", {})
            row_values = {
                feature: supplied.get(feature, defaults.get(feature, 0))
                for feature in features
            }
            row = pd.DataFrame([row_values], columns=features)

            try:
                probabilities = model.predict_proba(row)[0]
                classes = list(model.classes_)
                approved_index = classes.index(1) if 1 in classes else len(probabilities) - 1
                probability = float(probabilities[approved_index])
                predicted_class = int(model.predict(row)[0])

                st.markdown("---")
                st.subheader("Final assessment")
                dti = (
                    existing_debt / monthly_income
                ) * 100

                st.metric(
                    "Existing debt-to-income ratio",
                    f"{dti:.1f}%",
                )

                st.caption(
                    "Informational indicator based on existing monthly debt "
                    "and monthly income. It is not used by the trained model."
                )
                if predicted_class == 1:
                    st.success("MODEL RESULT: APPROVED PATTERN")
                else:
                    st.error("MODEL RESULT: REJECTED PATTERN")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Applicant",
                    applicant_name.strip(),
                )

                m2.metric(
                    "Approval probability",
                    f"{probability:.1%}",
                )

                m3.metric(
                    "Estimated outcome",
                    "Approved" if predicted_class == 1 else "Rejected",
                )

                st.progress(
                    min(1.0, max(0.0, probability))
                )

                st.info(
                    "This is a machine-learning research estimate, not a "
                    "real bank decision or guarantee."
                )
                st.caption(
                    "This probability describes patterns learned from the supplied research dataset. "
                    "It is not a bank decision, a guarantee, or an identity verification result."
                )

                st.markdown("#### Applicant summary")
                summary = pd.DataFrame(
                    [
                        ("Age", f"{age} years"),
                        ("Employment length", f"{employment_years:g} years"),
                        ("Monthly income", f"₹{monthly_income:,.0f}"),
                        ("Co-applicant monthly income", f"₹{coapplicant_income:,.0f}"),
                        ("Requested loan", f"₹{loan_amount:,.0f}"),
                        ("Loan term", f"{loan_term} years"),
                        ("Home ownership", home_ownership),
                        ("Loan purpose", loan_purpose),
                        ("Loan grade", loan_grade),
                        ("Interest rate", f"{interest_rate:.2f}%"),
                        ("Credit history length", f"{credit_history_years:g} years"),
                        ("Previous default on file", previous_default),
                        ("Existing monthly debt payments", f"₹{existing_debt:,.0f}"),
                        ("PAN upload", "Received; not verified"),
                    ],
                    columns=["Field", "Entered value"],
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

                st.markdown("#### Model explanation")
                try:
                    import shap

                    transformed = model.named_steps["preprocessor"].transform(row)
                    explainer = shap.TreeExplainer(model.named_steps["classifier"])
                    shap_values = explainer.shap_values(transformed)
                    if isinstance(shap_values, list):
                        contributions = np.asarray(shap_values[1])[0]
                    else:
                        shap_values = np.asarray(shap_values)
                        if shap_values.ndim == 3 and shap_values.shape[-1] == 2:
                            contributions = shap_values[0, :, 1]
                        elif shap_values.ndim == 3:
                            contributions = shap_values[0, :, 0]
                        else:
                            contributions = shap_values[0]
                    try:
                        names = list(model.named_steps["preprocessor"].get_feature_names_out())
                    except Exception:
                        names = [f"Feature {i + 1}" for i in range(len(contributions))]
                    explanation = pd.DataFrame(
                        {"Feature": names, "SHAP contribution": contributions}
                    )
                    explanation["Direction"] = np.where(
                        explanation["SHAP contribution"] >= 0,
                        "Pushes toward approval class",
                        "Pushes away from approval class",
                    )
                    explanation["Magnitude"] = explanation["SHAP contribution"].abs()
                    explanation = explanation.sort_values("Magnitude", ascending=False).head(8)
                    st.subheader("Individual SHAP feature impact")

                    shap_fig = px.bar(
                        explanation.sort_values(
                            "SHAP contribution", ascending=True
                        ),
                        x="SHAP contribution",
                        y="Feature",
                        color="Direction",
                        orientation="h",
                        color_discrete_map={
                            "Pushes toward approval class": "#16A34A",
                            "Pushes away from approval class": "#DC2626",
                        },
                        title="Top 8 factors influencing this prediction",
                    )

                    shap_fig.update_layout(
                        paper_bgcolor="#0B1120",
                        plot_bgcolor="#0B1120",
                        font_color="#E5E7EB",
                        yaxis_title="",
                        xaxis_title="SHAP contribution",
                        legend_title="Impact direction",
                    )

                    st.plotly_chart(
                        shap_fig,
                        use_container_width=True,
                    )

                    st.caption(
                        "Positive SHAP contributions push toward the approval "
                        "class; negative contributions push away from it. "
                        "These explain model behaviour, not causation."
                    )
                except Exception as exc:
                    st.info(
                        f"Local SHAP explanation unavailable: {exc}. "
                        "See Model Report for global importance."
                    )
                        # Gemini AI explanation
                st.markdown("#### Gemini AI explanation")
                st.info(
                    "Important: applicant name, age, employment length, co-applicant income, "
                    "home ownership, loan purpose, loan grade, interest rate, credit-history length, "
                    "previous default and existing debt are shown in the report but are not used by "
                    "this trained model unless they are present in its training feature list. "
                    "They do not change this prediction in the current version."
                )
                if "explanation" in locals():
                 shap_factors = explanation[
                    ["Feature", "SHAP contribution", "Direction"]
                ].to_dict(orient="records")
                else:
                 shap_factors = [
                    "SHAP explanation unavailable for this assessment."
                ]
                with st.spinner("Generating AI explanation..."):
                 ai_explanation = generate_loan_explanation(
                    prediction=(
                        "Approved" if predicted_class == 1 else "Rejected"
                    ),
                    approval_probability=probability,
                    shap_factors=shap_factors,
                )

                 st.markdown(ai_explanation)
                
            except Exception as exc:
                st.error(f"Could not calculate the assessment: {exc}")
                st.info("Check that the model files were trained from the dataset in this project.")

elif page == "Model Report":
    model, meta = assets()
    st.subheader("Loan approval distribution")

    loan_data = pd.read_csv(DATA)
    loan_data.columns = [
        str(c).strip().lower()
        for c in loan_data.columns
    ]

    loan_data["loan_status"] = (
        loan_data["loan_status"].astype(str).str.strip().str.title()
    )

    distribution = (
        loan_data["loan_status"]
        .value_counts()
        .reindex(["Approved", "Rejected"], fill_value=0)
        .reset_index()
    )

    distribution.columns = ["Loan Status", "Applications"]

    fig = px.pie(
        distribution,
        names="Loan Status",
        values="Applications",
        hole=0.55,
        title="Approved vs Rejected Applications",
        color="Loan Status",
        color_discrete_map={
            "Approved": "#16A34A",
            "Rejected": "#DC2626",
        },
    )

    fig.update_traces(
        textposition="inside",
        textinfo="label+percent",
    )

    fig.update_layout(
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        font_color="#E5E7EB",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Held-out model evaluation")
    metrics = meta["metrics"]
    cols = st.columns(4)
    for col, key, name in zip(
        cols,
        ["accuracy", "precision_approved", "recall_approved", "f1_approved"],
        ["Accuracy", "Precision (approved)", "Recall (approved)", "F1 (approved)"],
    ):
        col.metric(name, f"{metrics[key]:.3f}")
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "N/A")
    col2.metric("Selected model", meta["model_name"])
    st.write(
        f"Rows: {meta['dataset_rows']:,} · Train: {meta['training_rows']:,} · Test: {meta['test_rows']:,}"
    )
    st.subheader("Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            meta["confusion_matrix"],
            index=["Actual rejected", "Actual approved"],
            columns=["Predicted rejected", "Predicted approved"],
        ),
        use_container_width=True,
    )
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(meta["model_comparison"]), use_container_width=True, hide_index=True)
    importance_path = ROOT / "approval_feature_importance.csv"

    if importance_path.exists():
        importance = pd.read_csv(
            importance_path,
            header=None,
            names=["feature", "importance"],
        )

        importance["importance"] = pd.to_numeric(
            importance["importance"], errors="coerce"
        )

        importance = importance.dropna()
        importance = importance.sort_values(
            "importance", ascending=False
        ).head(15)

        st.subheader("Global feature importance")
        st.caption(
            "Shows which input features contribute most to the trained model's "
            "overall predictions. Importance does not establish causation."
        )

        fig = px.bar(
            importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Important Features",
            labels={
                "importance": "Importance",
                "feature": "Model feature",
            },
        )

        fig.update_layout(
            paper_bgcolor="#0B1120",
            plot_bgcolor="#0B1120",
            font_color="#E5E7EB",
            yaxis_title="",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Feature importance file not found.")

    st.caption(
        "Metrics are dataset-specific; they do not establish real-world validity, fairness, or causality."
    )

elif page == "Fairness Audit":
    df = data()
    if df is None:
        st.error("Dataset missing. See README.md.")
        st.stop()
    if "loan_status" not in df.columns:
        st.error("Target column loan_status not found.")
        st.stop()
    df["loan_status"] = df["loan_status"].map(target_value)
    df = df.dropna(subset=["loan_status"]).copy()
    df["loan_status"] = df["loan_status"].astype(int)
    model, meta = assets()
    groups = [
        column
        for column in ["gender", "education", "self_employed", "property_area", "married", "credit_history"]
        if column in df.columns
    ]
    if not groups:
        st.warning(
            "This dataset does not contain supported grouping attributes. Do not claim a comprehensive fairness audit."
        )
        st.stop()
    group = st.selectbox("Compare groups by", groups)
    rows = []
    for name, group_df in df.groupby(group, dropna=False, observed=False):
        actual = group_df.loan_status
        predicted = group_df.predicted_approved
        negatives = max(1, int((actual == 0).sum()))
        positives = max(1, int((actual == 1).sum()))
        rows.append(
            {
                "Group": str(name),
                "Applicants": len(group_df),
                "Observed approval (%)": round(actual.mean() * 100, 2),
                "Predicted approval (%)": round(predicted.mean() * 100, 2),
                "False approval rate (%)": round((((predicted == 1) & (actual == 0)).sum() / negatives) * 100, 2),
                "Missed approval rate (%)": round((((predicted == 0) & (actual == 1)).sum() / positives) * 100, 2),
            }
        )
    report = pd.DataFrame(rows)
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.subheader("Fairness disparity indicators")

    approval_gap = (
        report["Predicted approval (%)"].max()
        - report["Predicted approval (%)"].min()
    )

    false_approval_gap = (
        report["False approval rate (%)"].max()
        - report["False approval rate (%)"].min()
    )

    missed_approval_gap = (
        report["Missed approval rate (%)"].max()
        - report["Missed approval rate (%)"].min()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Approval rate gap", f"{approval_gap:.2f} pp")
    col2.metric("False approval gap", f"{false_approval_gap:.2f} pp")
    col3.metric("Missed approval gap", f"{missed_approval_gap:.2f} pp")

    st.caption(
        "pp = percentage points. These are descriptive group differences, "
        "not proof of discrimination or a legal fairness assessment."
    )
    st.bar_chart(report.set_index("Group")[["Observed approval (%)", "Predicted approval (%)"]])
    st.subheader("Prediction error comparison")

    error_comparison = report.melt(
        id_vars=["Group"],
        value_vars=[
            "False approval rate (%)",
            "Missed approval rate (%)",
        ],
        var_name="Error type",
        value_name="Rate (%)",
    )

    error_fig = px.bar(
        error_comparison,
        x="Group",
        y="Rate (%)",
        color="Error type",
        barmode="group",
        title="False approvals vs missed approvals",
        color_discrete_map={
            "False approval rate (%)": "#DC2626",
            "Missed approval rate (%)": "#F59E0B",
        },
    )

    error_fig.update_layout(
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        font_color="#E5E7EB",
        yaxis_title="Rate (%)",
        xaxis_title="Applicant group",
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True,
    )

    st.caption(
        "These error rates are descriptive comparisons. Small group sizes "
        "can make estimates unstable; they are not proof of discrimination."
    )
    st.caption(
        "Group differences are descriptive signals for investigation, not proof of discrimination. "
        "Interpret sample sizes, data quality, policy context and legal requirements."
    )

else:
    st.subheader("About Fairlens")

    st.markdown(
        """
        ### Explainable Loan Approval Research Prototype

        Fairlens is an educational machine-learning application designed
        to demonstrate loan approval prediction, model explainability,
        and descriptive fairness analysis.

        ### System architecture

        1. Applicant information is collected through a Streamlit interface.
        2. Input values are mapped to the trained model's 11 features.
        3. A tuned Random Forest pipeline generates an approval estimate.
        4. SHAP explains the model's individual prediction.
        5. Gemini generates a natural-language explanation.
        6. The Fairness Audit compares available applicant groups.

        ### Technology stack

        - Python
        - Streamlit
        - Pandas and NumPy
        - Scikit-learn
        - Random Forest
        - SHAP
        - Plotly
        - Google Gemini API

        ### Limitations

        - Historical training data may contain existing biases.
        - SHAP explains model behaviour, not causation.
        - Gemini explanations are generated text and may contain errors.
        - Group disparities do not independently establish discrimination.
        - Results may not generalize to other lenders or populations.
        - PAN uploads are not identity or authenticity verified.
        - Additional applicant fields may be collected but are not necessarily model inputs.
        - This prototype is not suitable for real lending decisions.

        ### Privacy and security

        Demo credentials and API configuration are stored locally in
        environment variables. This prototype does not implement
        production-grade authentication or regulated financial-data security.
        """
    )

    st.subheader("Run Fairlens locally")

    st.code(
        "python -m pip install -r requirements.txt\n"
        "python train_approval_model.py\n"
        "python -m streamlit run app.py",
        language="powershell",
    )

    st.caption(
        "Fairlens 2.0 | Deviathon 2.0 | Educational research prototype"
    )