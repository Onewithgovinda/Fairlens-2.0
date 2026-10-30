from pathlib import Path
import json, re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"loan_approval_dataset.csv"
if not DATA.exists():
    raise SystemExit("Dataset missing: put a documented approval dataset at data/loan_approval_dataset.csv. See README.md.")
df=pd.read_csv(DATA)
def clean(c): return re.sub(r"_+","_",re.sub(r"[^a-z0-9]+","_",str(c).strip().lower())).strip("_")
df.columns=[clean(c) for c in df.columns]
if "loan_status" not in df.columns:
    for alt in ["loanstatus","loan_approval","approved","loan_approved"]:
        if alt in df.columns: df=df.rename(columns={alt:"loan_status"}); break
if "loan_status" not in df.columns:
    raise SystemExit(f"Target loan_status not found. Columns: {df.columns.tolist()}")
def enc(v):
    s=str(v).strip().lower()
    if s in {"1","yes","y","approved","approve","true"}: return 1
    if s in {"0","no","n","rejected","reject","false"}: return 0
    try:
        n=float(s)
        return int(n) if n in (0,1) else np.nan
    except ValueError: return np.nan
df["loan_status"]=df["loan_status"].map(enc)
df=df.dropna(subset=["loan_status"]).copy()
df["loan_status"]=df["loan_status"].astype(int)
if len(df)<100: print(f"WARNING: only {len(df)} labelled rows; results are preliminary.")
if df["loan_status"].nunique()!=2 or df["loan_status"].value_counts().min()<2:
    raise SystemExit("Need at least two classes and at least two examples per class.")
drop=["loan_status"]+[c for c in ["loan_id","id","unnamed_0"] if c in df.columns]
X=df.drop(columns=drop); y=df["loan_status"]
num=X.select_dtypes(include=np.number).columns.tolist()
cat=[c for c in X.columns if c not in num]
prep=ColumnTransformer([
 ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
# Reserve a final test set, and use a separate validation split to select the model.
Xdev,Xte,ydev,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
Xtr,Xval,ytr,yval=train_test_split(Xdev,ydev,test_size=.25,random_state=43,stratify=ydev)
models={
 "Logistic Regression baseline":LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42),
 "Random Forest":RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1)
}
pipes={}; results=[]
for name,clf in models.items():
    pipe=Pipeline([("preprocessor",prep),("classifier",clf)])
    pipe.fit(Xtr,ytr); pred_val=pipe.predict(Xval); prob_val=pipe.predict_proba(Xval)[:,1]
    try: auc_val=float(roc_auc_score(yval,prob_val))
    except ValueError: auc_val=None
    results.append({"model":name,"validation_accuracy":float(accuracy_score(yval,pred_val)),"validation_precision_approved":float(precision_score(yval,pred_val,zero_division=0)),"validation_recall_approved":float(recall_score(yval,pred_val,zero_division=0)),"validation_f1_approved":float(f1_score(yval,pred_val,zero_division=0)),"validation_roc_auc":auc_val})
    pipes[name]=pipe
best_val=max(results,key=lambda x:x["validation_f1_approved"]); name=best_val["model"]
# Refit the selected model on train + validation, then evaluate exactly once on the held-out test set.
Xfit=pd.concat([Xtr,Xval]); yfit=pd.concat([ytr,yval])
clf=models[name]
model=Pipeline([("preprocessor",prep),("classifier",clf)])
model.fit(Xfit,yfit)
pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
try: test_auc=float(roc_auc_score(yte,prob))
except ValueError: test_auc=None
best={"accuracy":float(accuracy_score(yte,pred)),"precision_approved":float(precision_score(yte,pred,zero_division=0)),"recall_approved":float(recall_score(yte,pred,zero_division=0)),"f1_approved":float(f1_score(yte,pred,zero_division=0)),"roc_auc":test_auc}
cm=confusion_matrix(yte,pred,labels=[0,1]).tolist()
joblib.dump(model,ROOT/"approval_model.joblib")
perm=permutation_importance(model,Xte,yte,n_repeats=5,random_state=42,scoring="f1",n_jobs=-1)
pd.DataFrame({"feature":Xte.columns,"importance":perm.importances_mean}).sort_values("importance",ascending=False).to_csv(ROOT/"approval_feature_importance.csv",index=False)
defaults={}; categories={}
for c in X.columns:
    if c in cat:
        vals=X[c].dropna().astype(str).value_counts().index.tolist()
        categories[c]=vals[:50]; defaults[c]=vals[0] if vals else "Unknown"
    else:
        v=pd.to_numeric(X[c],errors="coerce").median(); defaults[c]=float(v) if pd.notna(v) else 0.0
meta={"dataset_rows":int(len(df)),"training_rows":int(len(Xfit)),"validation_rows":int(len(Xval)),"test_rows":int(len(Xte)),"target_column":"loan_status","model_name":name,"feature_columns":list(X.columns),"numeric_columns":num,"categorical_columns":cat,"category_values":categories,"form_defaults":defaults,"metrics":best,"model_comparison":results,"confusion_matrix":cm,"classification_report":classification_report(yte,pred,target_names=["Rejected","Approved"],output_dict=True,zero_division=0),"random_state":42,"test_size":.2}
(ROOT/"approval_metrics.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print("Validation comparison (used for model selection):")
print(pd.DataFrame(results).to_string(index=False))
print("\\nFinal held-out test metrics:")
print(best)
print(f"\nSelected model: {name}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(np.array(cm))
print("Saved approval_model.joblib, approval_metrics.json and approval_feature_importance.csv")
print("Model selection used validation data; final metrics use a separate held-out test set. Results do not prove real-world validity or fairness.")
