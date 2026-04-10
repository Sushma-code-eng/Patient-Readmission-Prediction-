
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, json, pickle

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, precision_recall_curve,
                              average_precision_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150

PROJ = "/home/claude/patient-readmission-prediction"
VIZ = PROJ + "/visualizations"

print("=" * 70)
print("  PATIENT READMISSION PREDICTION PIPELINE")
print("=" * 70)

df = pd.read_csv(PROJ + "/data/patient_readmission.csv")
print("Dataset: {} rows x {} columns".format(df.shape[0], df.shape[1]))
print("Readmission rate: {:.1f}%".format(df["readmitted"].mean()*100))

colors = ["#4CAF50", "#F44336"]

# 1. Target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["readmitted"].value_counts().plot.pie(ax=axes[0], colors=colors, autopct="%1.1f%%",
    labels=["Not Readmitted", "Readmitted"], startangle=90, explode=(0, 0.05))
axes[0].set_title("Readmission Distribution", fontweight="bold"); axes[0].set_ylabel("")
df["readmitted"].value_counts().plot.bar(ax=axes[1], color=colors, edgecolor="white")
axes[1].set_title("Readmission Counts", fontweight="bold")
axes[1].set_xticklabels(["Not Readmitted", "Readmitted"], rotation=0)
plt.tight_layout(); plt.savefig(VIZ+"/01_target_distribution.png", bbox_inches="tight"); plt.close()
print("Saved: 01")

# 2. Age
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
age_order = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
age_readmit = df.groupby("age_group")["readmitted"].mean().reindex(age_order) * 100
axes[0].bar(range(len(age_readmit)), age_readmit.values, color="#2196F3")
axes[0].set_xticks(range(len(age_readmit))); axes[0].set_xticklabels(age_order, rotation=45, ha="right")
axes[0].set_title("Readmission Rate by Age", fontweight="bold"); axes[0].set_ylabel("Rate (%)")
axes[0].axhline(y=df["readmitted"].mean()*100, color="red", linestyle="--")
age_counts = df.groupby("age_group").size().reindex(age_order)
axes[1].bar(range(len(age_counts)), age_counts.values, color="#FF9800")
axes[1].set_xticks(range(len(age_counts))); axes[1].set_xticklabels(age_order, rotation=45, ha="right")
axes[1].set_title("Patient Count by Age", fontweight="bold")
plt.tight_layout(); plt.savefig(VIZ+"/02_age_analysis.png", bbox_inches="tight"); plt.close()
print("Saved: 02")

# 3. Clinical
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, col, color in zip(axes[0], ["admission_type","discharge_disposition","diag_category"], ["#E91E63","#9C27B0","#FF5722"]):
    rates = df.groupby(col)["readmitted"].mean().sort_values(ascending=False) * 100
    ax.barh(rates.index, rates.values, color=color); ax.set_title("By "+col, fontweight="bold")
for ax, col in zip(axes[1], ["time_in_hospital","number_inpatient","number_diagnoses"]):
    sns.boxplot(data=df, x="readmitted", y=col, ax=ax, palette=colors, width=0.5)
    ax.set_title(col, fontweight="bold"); ax.set_xticklabels(["No","Yes"])
plt.tight_layout(); plt.savefig(VIZ+"/03_clinical_features.png", bbox_inches="tight"); plt.close()
print("Saved: 03")

# 4. Medications
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col, color in zip(axes, ["insulin","A1Cresult","diabetesMed"], ["#00BCD4","#FF9800","#8BC34A"]):
    rates = df.groupby(col)["readmitted"].mean().sort_values(ascending=False) * 100
    ax.bar(rates.index, rates.values, color=color); ax.set_title("By "+col, fontweight="bold")
    ax.axhline(y=df["readmitted"].mean()*100, color="red", linestyle="--")
plt.tight_layout(); plt.savefig(VIZ+"/04_medication_analysis.png", bbox_inches="tight"); plt.close()
print("Saved: 04")

# 5. Correlation
df_enc = df.copy()
for col in df_enc.select_dtypes(include="object").columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
fig, ax = plt.subplots(figsize=(14, 10))
corr = df_enc.corr(); mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax, annot_kws={"size":8})
ax.set_title("Correlation Matrix", fontweight="bold")
plt.tight_layout(); plt.savefig(VIZ+"/05_correlation_heatmap.png", bbox_inches="tight"); plt.close()
print("Saved: 05")

# FEATURE ENGINEERING
df_ml = df_enc.copy()
df_ml["total_visits"] = df_ml["number_outpatient"] + df_ml["number_emergency"] + df_ml["number_inpatient"]
df_ml["emergency_ratio"] = df_ml["number_emergency"] / (df_ml["total_visits"] + 1)
df_ml["inpatient_ratio"] = df_ml["number_inpatient"] / (df_ml["total_visits"] + 1)
df_ml["med_per_day"] = df_ml["num_medications"] / (df_ml["time_in_hospital"] + 1)
df_ml["proc_per_day"] = df_ml["num_procedures"] / (df_ml["time_in_hospital"] + 1)
df_ml["lab_per_day"] = df_ml["num_lab_procedures"] / (df_ml["time_in_hospital"] + 1)
df_ml["complexity_score"] = df_ml["number_diagnoses"] + df_ml["num_procedures"] + df_ml["num_medications"]/5
df_ml["high_risk_flag"] = ((df_ml["number_inpatient"]>1)|(df_ml["number_emergency"]>1)|(df_ml["time_in_hospital"]>7)).astype(int)
print("Features: {}".format(df_ml.shape[1]-1))

# MODEL BUILDING
X = df_ml.drop("readmitted", axis=1); y = df_ml["readmitted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_sm); X_tsc = scaler.transform(X_test)
print("Train: {} | Test: {} | After SMOTE: {}".format(len(X_train), len(X_test), len(X_sm)))

spw = len(y_train[y_train==0])/max(len(y_train[y_train==1]),1)
models = {
    "Logistic Regression": ("s", LogisticRegression(max_iter=1000, random_state=42)),
    "Decision Tree": ("m", DecisionTreeClassifier(max_depth=8, random_state=42)),
    "Random Forest": ("m", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    "Gradient Boosting": ("m", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
    "XGBoost": ("o", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=spw, random_state=42, verbosity=0, n_jobs=-1, eval_metric="logloss"))
}

results = {}
header = "{:<25} {:>7} {:>7} {:>7} {:>7} {:>7}".format("Model","Acc","Prec","Rec","F1","AUC")
print("\n" + header)
print("-"*60)
for name, (t, model) in models.items():
    if t=="s": model.fit(X_sc, y_sm); yp=model.predict(X_tsc); ypr=model.predict_proba(X_tsc)[:,1]
    elif t=="o": model.fit(X_train, y_train); yp=model.predict(X_test); ypr=model.predict_proba(X_test)[:,1]
    else: model.fit(X_sm, y_sm); yp=model.predict(X_test); ypr=model.predict_proba(X_test)[:,1]
    r = {"accuracy":accuracy_score(y_test,yp), "precision":precision_score(y_test,yp),
         "recall":recall_score(y_test,yp), "f1":f1_score(y_test,yp),
         "auc":roc_auc_score(y_test,ypr), "y_pred":yp, "y_prob":ypr}
    results[name] = r
    print("{:<25} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(name, r["accuracy"], r["precision"], r["recall"], r["f1"], r["auc"]))

bn = max(results, key=lambda k: results[k]["auc"])
bm = models[bn][1]
print("\nBest: {} (AUC={:.4f})".format(bn, results[bn]["auc"]))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvs = cross_val_score(bm, X, y, cv=cv, scoring="roc_auc")
print("CV AUC: {:.4f} +/- {:.4f}".format(cvs.mean(), cvs.std()))
print("\n" + classification_report(y_test, results[bn]["y_pred"], target_names=["Not Readmitted","Readmitted"]))

pickle.dump(bm, open(PROJ+"/models/best_model.pkl","wb"))
pickle.dump(scaler, open(PROJ+"/models/scaler.pkl","wb"))

yp_f = results[bn]["y_prob"]; ypd_f = results[bn]["y_pred"]

# 6. ROC + PR
fig, axes = plt.subplots(1,2,figsize=(16,6))
croc = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"]
for i,(name,res) in enumerate(results.items()):
    fpr,tpr,_ = roc_curve(y_test, res["y_prob"])
    axes[0].plot(fpr,tpr,color=croc[i],lw=2,label="{} ({:.3f})".format(name,res["auc"]))
axes[0].plot([0,1],[0,1],"k--"); axes[0].set_title("ROC Curves",fontweight="bold")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend(fontsize=9)
for i,(name,res) in enumerate(results.items()):
    p,r,_ = precision_recall_curve(y_test,res["y_prob"])
    ap = average_precision_score(y_test,res["y_prob"])
    axes[1].plot(r,p,color=croc[i],lw=2,label="{} ({:.3f})".format(name,ap))
axes[1].set_title("PR Curves",fontweight="bold"); axes[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(VIZ+"/06_roc_pr_curves.png",bbox_inches="tight"); plt.close()
print("Saved: 06")

# 7. Confusion Matrix
fig, axes = plt.subplots(1,2,figsize=(14,5))
cm = confusion_matrix(y_test, ypd_f)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=["No","Yes"], yticklabels=["No","Yes"])
axes[0].set_title("Confusion Matrix",fontweight="bold"); axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")
cmn = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
sns.heatmap(cmn, annot=True, fmt=".2%", cmap="Oranges", ax=axes[1], xticklabels=["No","Yes"], yticklabels=["No","Yes"])
axes[1].set_title("Normalized",fontweight="bold"); axes[1].set_ylabel("Actual"); axes[1].set_xlabel("Predicted")
plt.tight_layout(); plt.savefig(VIZ+"/07_confusion_matrix.png",bbox_inches="tight"); plt.close()
print("Saved: 07")

# 8. Model Comparison
fig, axes = plt.subplots(1,2,figsize=(16,6))
mn = list(results.keys()); x = np.arange(len(mn)); w=0.2
for i,(m,k) in enumerate([("AUC","auc"),("F1","f1"),("Recall","recall"),("Prec","precision")]):
    axes[0].bar(x+i*w, [results[n][k] for n in mn], w, label=m)
axes[0].set_xticks(x+1.5*w); axes[0].set_xticklabels(mn,rotation=25,ha="right",fontsize=9)
axes[0].set_title("Model Comparison",fontweight="bold"); axes[0].set_ylim(0,1); axes[0].legend()
av = [results[n]["auc"] for n in mn]
axes[1].barh(mn, av, color=croc); axes[1].set_title("AUC Comparison",fontweight="bold"); axes[1].set_xlim(0.5,1)
for i,v in enumerate(av): axes[1].text(v+0.005,i,"{:.3f}".format(v),va="center")
plt.tight_layout(); plt.savefig(VIZ+"/08_model_comparison.png",bbox_inches="tight"); plt.close()
print("Saved: 08")

# 9. Probability Distribution
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(yp_f[y_test==0],bins=40,alpha=0.6,color="#4CAF50",label="Not Readmitted",density=True)
ax.hist(yp_f[y_test==1],bins=40,alpha=0.6,color="#F44336",label="Readmitted",density=True)
ax.set_title("Predicted Probability Distribution",fontweight="bold")
ax.axvline(x=0.5,color="black",linestyle="--"); ax.legend()
plt.tight_layout(); plt.savefig(VIZ+"/09_probability_distribution.png",bbox_inches="tight"); plt.close()
print("Saved: 09")

# SHAP (always use XGBoost for tree-based explainability)
shap_model = models["XGBoost"][1]
explainer = shap.TreeExplainer(shap_model)
sv = explainer.shap_values(X_test)
fig=plt.figure(figsize=(12,8)); shap.summary_plot(sv,X_test,show=False,max_display=20)
plt.title("SHAP Impact on Readmission",fontweight="bold")
plt.tight_layout(); plt.savefig(VIZ+"/10_shap_summary.png",bbox_inches="tight"); plt.close()
print("Saved: 10")

fig=plt.figure(figsize=(10,7)); shap.summary_plot(sv,X_test,plot_type="bar",show=False,max_display=20)
plt.title("Mean |SHAP| Importance",fontweight="bold")
plt.tight_layout(); plt.savefig(VIZ+"/11_shap_bar.png",bbox_inches="tight"); plt.close()
print("Saved: 11")

imp = pd.DataFrame({"feature":X.columns,"importance":shap_model.feature_importances_}).sort_values("importance",ascending=False)
print("\nTop 15 Features:")
for _,r in imp.head(15).iterrows():
    print("  {:<25s} {:.4f}".format(r["feature"], r["importance"]))

# RISK STRATIFICATION
rdf = pd.DataFrame({"actual":y_test.values,"prob":yp_f})
rdf["tier"] = pd.cut(rdf["prob"],bins=[0,0.20,0.40,0.60,1.0],labels=["Low","Medium","High","Very High"])
print("\n{:<12} {:>6} {:>8}".format("Tier","Count","Rate"))
for t in ["Low","Medium","High","Very High"]:
    s=rdf[rdf["tier"]==t]
    if len(s)>0: print("{:<12} {:>6} {:>7.1f}%".format(t, len(s), s["actual"].mean()*100))

fig, axes = plt.subplots(1,2,figsize=(14,5))
tc = ["#4CAF50","#FFC107","#FF9800","#F44336"]
tco = rdf["tier"].value_counts().reindex(["Low","Medium","High","Very High"]).fillna(0)
axes[0].bar(tco.index,tco.values,color=tc); axes[0].set_title("Patients by Risk Tier",fontweight="bold")
tr = rdf.groupby("tier")["actual"].mean().reindex(["Low","Medium","High","Very High"]).fillna(0)*100
axes[1].bar(tr.index,tr.values,color=tc); axes[1].set_title("Readmission Rate by Tier",fontweight="bold")
for i,v in enumerate(tr.values): axes[1].text(i,v+0.5,"{:.1f}%".format(v),ha="center",fontweight="bold")
plt.tight_layout(); plt.savefig(VIZ+"/12_risk_stratification.png",bbox_inches="tight"); plt.close()
print("Saved: 12")

summary = {"best_model":bn,"auc":round(results[bn]["auc"],4),"f1":round(results[bn]["f1"],4),
    "recall":round(results[bn]["recall"],4),"precision":round(results[bn]["precision"],4),
    "cv_auc":round(cvs.mean(),4),"readmission_rate":round(df["readmitted"].mean(),4),
    "top_features":imp.head(15)[["feature","importance"]].to_dict("records")}
json.dump(summary, open(PROJ+"/reports/model_summary.json","w"), indent=2)

print("\n" + "="*70)
print("  PIPELINE COMPLETE!")
print("="*70)
