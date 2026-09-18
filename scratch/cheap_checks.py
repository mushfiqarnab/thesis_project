import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

u = pd.read_csv("data/csv/multimodal_10k_unbiased.csv")
b = pd.read_csv("data/csv/multimodal_10k.csv")

print(u.image_path.head(3).tolist(), u.mask_path.head(3).tolist())
print("image_path shared u and b:", len(set(u.image_path) & set(b.image_path)))
print("unique mask_path u/b:", u.mask_path.nunique(), b.mask_path.nunique())
for n, d in (("unbiased", u), ("biased", b)):
    print(n, "corr(scar,threat):", round(d.scar.corr(d.threat), 3))
    print(d.groupby("subject")[["scar", "threat"]].mean().round(2).T)

# what does physiology alone predict on held-out subjects?
X, y, g = u[["hrv", "gsr"]].to_numpy(), u.threat.to_numpy(), u.subject.to_numpy()
for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    m = LogisticRegression(max_iter=1000).fit((X[tr]-mu)/sd, y[tr])
    print(k, sorted(set(g[te])), round(m.score((X[te]-mu)/sd, y[te]), 3),
          "majority", round(max(y[te].mean(), 1-y[te].mean()), 3))
