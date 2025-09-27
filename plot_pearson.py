import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns


path = r"D:\Desktop\686p\results\analysis\pred_noref_notrain.csv"
df = pd.read_csv(path, index_col=False)
df["correct"] = df["correct"].astype(int)


metrics = ["eu", "mean", "std", "median", "q25", "q75"]

corr_matrix = pd.DataFrame(index=["r", "p"], columns=metrics, dtype=float)
for m in metrics:
    r, p = stats.pearsonr(df["correct"], df[m])
    corr_matrix.loc["r", m] = r
    corr_matrix.loc["p", m] = p


print(corr_matrix)


plt.figure()
r_vals = corr_matrix.loc["r"].to_frame().T
sns.heatmap(
    r_vals, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False, linewidths=0.5
)
for i, m in enumerate(metrics):
    p = corr_matrix.loc["p", m]
    if p < 0.001:
        star = "***\n"
    elif p < 0.01:
        star = "**\n"
    elif p < 0.05:
        star = "*\n"
    else:
        star = "\n"
    plt.text(i + 0.5, 0.5, star, ha="center", va="center", color="black", fontsize=14)

plt.yticks([])
plt.title("P(correct) vs metrics - Pearson r\n(* p<0.05, ** p<0.01, *** p<0.001)")
plt.tight_layout()
plt.show()


for m in metrics:
    g = sns.lmplot(
        x=m,
        y="correct",
        data=df,
        logistic=True,
        ci=95,
        scatter_kws={"alpha": 0.3, "s": 20},
    )
    g.set_axis_labels(m, "P(correct=1)")
    plt.title(
        f'P(correct) vs {m} - r={corr_matrix.loc["r",m]:.2f}, p={corr_matrix.loc["p",m]:.3f}'
    )
    plt.tight_layout()
    plt.show()
