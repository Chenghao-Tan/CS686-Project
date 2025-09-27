import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split


path = r"D:\Desktop\686p\results\analysis\pred_noref_notrain.csv"
df = pd.read_csv(path, index_col=False)
df["correct"] = df["correct"].astype(int)


X = df[["eu"]].values  # (n_samples,1)
y = df["correct"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression(solver="lbfgs")
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)[:, 1]
fpr, tpr, th = roc_curve(y_test, y_score)

roc_auc = auc(fpr, tpr)
print(f"AUC = {roc_auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for single-feature (eu) classifier")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(
    [X_test[y_test == 0].flatten(), X_test[y_test == 1].flatten()],
    bins=30,
    label=["incorrect", "correct"],
    alpha=0.6,
    density=True,
)
plt.xlabel("eu")
plt.ylabel("Density")
plt.title("Distribution of eu by class")
plt.legend()
plt.tight_layout()
plt.show()
