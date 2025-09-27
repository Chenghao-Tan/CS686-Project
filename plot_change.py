import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


path = r"D:\Desktop\686p\results\analysis\pred_noref_notrain.csv"
df1 = pd.read_csv(path, index_col=False)
df1["correct"] = df1["correct"].astype(int)
path = r"D:\Desktop\686p\results\analysis\pred_ref_notrain.csv"
df2 = pd.read_csv(path, index_col=False)
df2["correct"] = df2["correct"].astype(int)


df_list = [df1, df2]
labels = ["exp1", "exp2"]

metrics = [
    "eu",
    "logit",
    "logit_top",
    "prob",
    "prob_top",
    "mean",
    "std",
    "median",
    "q25",
    "q75",
]

for df, label in zip(df_list, labels):
    df["source"] = label
big_df = pd.concat(df_list, ignore_index=True)

summary = big_df.groupby("source")[["correct"] + metrics].mean().reset_index()
long_summary = summary.melt(id_vars="source", var_name="metric", value_name="avg")
plt.figure()
plt.ylabel("metric mean")
sns.lineplot(data=long_summary, x="metric", y="avg", hue="source", marker="o")
plt.xticks(rotation=45)
plt.title(f"{labels[0]} vs {labels[1]}: metric mean comparisons")
plt.tight_layout()
plt.show()

melted = big_df.melt(
    id_vars=["source"],
    value_vars=["correct"] + metrics,
    var_name="metric",
    value_name="value",
)
g = sns.catplot(
    data=melted,
    x="source",
    y="value",
    col="metric",
    kind="box",
    col_wrap=4,
    sharey=False,
    height=3,
    aspect=1,
)
g.figure.suptitle(f"{labels[0]} vs {labels[1]}: metric distributions (boxplot)", y=1.02)
plt.show()

g = sns.catplot(
    data=melted,
    x="source",
    y="value",
    col="metric",
    kind="violin",
    col_wrap=4,
    sharey=False,
    height=3,
    aspect=1,
)
g.figure.suptitle(
    f"{labels[0]} vs {labels[1]}: metric distributions (violinplot)", y=1.02
)
plt.show()

"""
for m in ["correct"] + metrics:
    plt.figure()
    sns.pointplot(data=big_df, x="source", y=m, capsize=0.1)
    plt.title(f"{m} 95% CI")
    plt.tight_layout()
    plt.show()
"""

summary = big_df.groupby("source")[metrics].mean().reset_index()
pivot = summary.set_index("source").T
baseline = pivot[labels[0]]
delta = pivot.subtract(baseline, axis=0)
pct_change = delta.div(baseline, axis=0) * 100
pct_change = pct_change.drop(columns=[labels[0]])
pct_change.plot(kind="bar", rot=45)
plt.axhline(0, color="gray", linewidth=1)
plt.ylabel(f"change (%)")
plt.title(f"{labels[1]} vs {labels[0]}: metric mean change")
plt.tight_layout()
plt.show()
