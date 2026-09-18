import json, hashlib, pandas as pd, numpy as np

u = pd.read_csv("data/csv/multimodal_10k_unbiased.csv")
b = pd.read_csv("data/csv/multimodal_10k.csv")

def get_hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

u_hash = get_hash(u)
b_hash = get_hash(b)

subjects = sorted(u.subject.unique())
rng = np.random.default_rng(0)
rng.shuffle(subjects)

folds = []
for k in range(5):
    test_subs = subjects[k*3 : (k+1)*3]
    rem_subs = [s for s in subjects if s not in test_subs]
    val_subs = rem_subs[:2]
    train_subs = rem_subs[2:]
    
    folds.append({
        "fold": k,
        "test": test_subs,
        "val": val_subs,
        "train": train_subs
    })

out = {
    "seed": 0,
    "u_hash": u_hash,
    "b_hash": b_hash,
    "folds": folds
}

# Assert disjointness for fold 0
f0 = folds[0]
assert set(f0["train"]).isdisjoint(f0["val"])
assert set(f0["train"]).isdisjoint(f0["test"])
assert set(f0["val"]).isdisjoint(f0["test"])

for d in [u, b]:
    tr_img = set(d[d.subject.isin(f0["train"])].image_path)
    val_img = set(d[d.subject.isin(f0["val"])].image_path)
    te_img = set(d[d.subject.isin(f0["test"])].image_path)
    assert tr_img.isdisjoint(val_img)
    assert tr_img.isdisjoint(te_img)
    assert val_img.isdisjoint(te_img)

with open("data/csv/folds_dry_run.json", "w") as f:
    json.dump(out, f, indent=2)

print("Folds prepared successfully.")
