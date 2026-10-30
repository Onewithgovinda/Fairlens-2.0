import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "approval_model.joblib"
METRICS = ROOT / "approval_metrics.json"
DATA = ROOT / "data" / "loan_approval_dataset.csv"

st.set_page_config(page_title="Fairlens | Explainable Loan Approval", page_icon="🏦", layout="wide")
st.markdown("""<style>
.stApp{background:#F7F7F2} section[data-testid="stSidebar"]{background:#173D32}
section[data-testid="stSidebar"] *{color:white!important}
.hero{background:#173D32;color:white;padding:1.4rem;border-radius:16px}
.hero h1{color:white;margin:0}.hero p{color:#E2EDE5;margin:0}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Fairlens</h1><p>Explainable Loan Approval Research Prototype</p></div>', unsafe_allow_html=True)
st.warning("Educational prototype only. This is not a bank decision or a reliable estimate for real lending.")

page = st.sidebar.radio("Navigate", ["Applicant Assessment", "Model Report", "Fairness Audit", "About & Limitations"])
st.sidebar.caption("DEV-P02 · Deviathon 2.0")

def assets():
    if not MODEL.exists() or not METRICS.exists():
        st.error("Model files missing. Follow README.md to train the model first.")
        st.stop()
    return joblib.load(MODEL), json.loads(METRICS.read_text(encoding="utf-8"))

def data():
    if not DATA.exists():
        return None
    d = pd.read_csv(DATA)
    d.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in d.columns]
    return d

def target_value(v):
    s = str(v).strip().lower()
    if s in {"1","yes","y","approved","approve","true"}: return 1
    if s in {"0","no","n","rejected","reject","false"}: return 0
    try:
        n=float(s)
        return int(n) if n in (0,1) else np.nan
    except ValueError:
        return np.nan

def label(s):
    return s.replace("_"," ").title()

if page == "Applicant Assessment":
    model, meta = assets()
    st.subheader("Applicant assessment")
    st.write("Enter a synthetic or appropriately permissioned sample application.")
    features=meta["feature_columns"]
    cats=set(meta["categorical_columns"])
    defaults=meta["form_defaults"]
    categories=meta["category_values"]
    values={}
    with st.form("assessment"):
        left,right=st.columns(2)
        for i,f in enumerate(features):
            col=left if i%2==0 else right
            if f in cats:
                opts=categories.get(f,[]) or ["Unknown"]
                default=defaults.get(f,opts[0])
                values[f]=col.selectbox(label(f),opts,index=opts.index(default) if default in opts else 0,key=f)
            else:
                v=float(defaults.get(f,0))
                step=max(1.0,abs(v)*0.05)
                values[f]=col.number_input(label(f),min_value=0.0,value=max(0.0,v),step=step,key=f)
        submitted=st.form_submit_button("Estimate and explain",type="primary",use_container_width=True)
    if submitted:
        row=pd.DataFrame([{f:values[f] for f in features}],columns=features)
        p=float(model.predict_proba(row)[0][1])
        c1,c2=st.columns(2)
        c1.metric("Estimated historical approval likelihood",f"{p:.1%}")
        c2.metric("Model class","Approved pattern" if p>=.5 else "Rejected pattern")
        st.progress(min(1,max(0,p)))
        st.caption("The 0.50 threshold is a model default, not a bank policy. Probability is not a guarantee.")
        st.subheader("Applicant-level explanation")
        try:
            import shap
            transformed=model.named_steps["preprocessor"].transform(row)
            explainer=shap.TreeExplainer(model.named_steps["classifier"])
            sv=explainer.shap_values(transformed)
            if isinstance(sv,list):
                vals=np.asarray(sv[1])[0]
            else:
                sv=np.asarray(sv)
                vals=sv[0,:,1] if sv.ndim==3 and sv.shape[-1]==2 else (sv[0,:,0] if sv.ndim==3 else sv[0])
            try: names=list(model.named_steps["preprocessor"].get_feature_names_out())
            except Exception: names=[f"Feature {i+1}" for i in range(len(vals))]
            exp=pd.DataFrame({"Feature":names,"SHAP contribution":vals})
            exp["Interpretation"]=np.where(exp["SHAP contribution"]>=0,"Pushes toward approval class","Pushes away from approval class")
            exp["Magnitude"]=exp["SHAP contribution"].abs()
            exp=exp.sort_values("Magnitude",ascending=False).head(8)
            st.dataframe(exp[["Feature","SHAP contribution","Interpretation"]],use_container_width=True,hide_index=True)
            st.caption("SHAP describes this model's behaviour, not causation or the actual reason a lender would decide.")
        except Exception as e:
            st.info(f"Local SHAP explanation unavailable: {e}. See Model Report for global importance.")
        st.info("Review inputs carefully. Do not use this output as the sole basis for a financial decision.")

elif page == "Model Report":
    model,meta=assets()
    st.subheader("Held-out model evaluation")
    m=meta["metrics"]
    cols=st.columns(4)
    for col,key,name in zip(cols,["accuracy","precision_approved","recall_approved","f1_approved"],["Accuracy","Precision (approved)","Recall (approved)","F1 (approved)"]):
        col.metric(name,f"{m[key]:.3f}")
    col1,col2=st.columns(2)
    col1.metric("ROC-AUC",f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A")
    col2.metric("Selected model",meta["model_name"])
    st.write(f"Rows: {meta['dataset_rows']:,} · Train: {meta['training_rows']:,} · Test: {meta['test_rows']:,}")
    st.subheader("Confusion matrix")
    st.dataframe(pd.DataFrame(meta["confusion_matrix"],index=["Actual rejected","Actual approved"],columns=["Predicted rejected","Predicted approved"]),use_container_width=True)
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(meta["model_comparison"]),use_container_width=True,hide_index=True)
    path=ROOT/"approval_feature_importance.csv"
    if path.exists():
        imp=pd.read_csv(path).head(15)
        st.subheader("Global permutation importance")
        st.dataframe(imp,use_container_width=True,hide_index=True)
        st.bar_chart(imp.set_index("feature")["importance"])
    st.caption("Metrics are dataset-specific; they do not establish real-world validity, fairness, or causality.")

elif page == "Fairness Audit":
    df=data()
    if df is None:
        st.error("Dataset missing. See README.md.")
        st.stop()
    if "loan_status" not in df.columns:
        st.error("Target column loan_status not found.")
        st.stop()
    df["loan_status"]=df["loan_status"].map(target_value)
    df=df.dropna(subset=["loan_status"]).copy()
    df["loan_status"]=df["loan_status"].astype(int)
    model,meta=assets()
    groups=[c for c in ["gender","education","self_employed","property_area","married","credit_history"] if c in df.columns]
    if not groups:
        st.warning("This dataset does not contain supported grouping attributes. Do not claim a comprehensive fairness audit.")
        st.stop()
    group=st.selectbox("Compare groups by",groups)
    X=df[meta["feature_columns"]]
    df["predicted_approved"]=model.predict(X)
    rows=[]
    for name,g in df.groupby(group,dropna=False,observed=False):
        actual=g.loan_status
        pred=g.predicted_approved
        negatives=max(1,int((actual==0).sum()))
        positives=max(1,int((actual==1).sum()))
        rows.append({"Group":str(name),"Applicants":len(g),"Observed approval (%)":round(actual.mean()*100,2),
          "Predicted approval (%)":round(pred.mean()*100,2),
          "False approval rate (%)":round((((pred==1)&(actual==0)).sum()/negatives)*100,2),
          "Missed approval rate (%)":round((((pred==0)&(actual==1)).sum()/positives)*100,2)})
    report=pd.DataFrame(rows)
    st.dataframe(report,use_container_width=True,hide_index=True)
    st.bar_chart(report.set_index("Group")[["Observed approval (%)","Predicted approval (%)"]])
    st.caption("Group differences are descriptive signals for investigation, not proof of discrimination. Interpret sample sizes, data quality, policy context and legal requirements.")

else:
    st.subheader("Method and limitations")
    st.markdown("""
**Workflow:** data validation → stratified holdout split → preprocessing pipeline → baseline/model comparison → held-out metrics → local/global explanations → group audit.

**Limitations**
- Historical labels can reflect historical policies and bias.
- SHAP explains model behaviour, not causality.
- Group disparities alone do not prove discrimination.
- Results may not transfer to another lender, population, or time period.
- This educational prototype must not be used for actual lending decisions.
""")
    st.code("python -m pip install -r requirements.txt\npython train_approval_model.py\npython -m streamlit run app.py",language="powershell")
