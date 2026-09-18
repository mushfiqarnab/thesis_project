import json
import pandas as pd

u = pd.read_csv("data/csv/multimodal_10k_unbiased.csv")
sp = json.load(open("data/csv/multimodal_10k_strict_split_seed42.json"))

tr, va = u.iloc[sp["train_idx"]], u.iloc[sp["val_idx"]]

print(list(u.columns))
print("train subj:", sorted(tr["subject"].unique()))
print("val subj:  ", sorted(va["subject"].unique()))

IMG = "image_path"
LAB = "threat"
SCAR = "scar"

print("unique images:", u[IMG].nunique(), "| train_val images overlap:", len(set(tr[IMG]) & set(va[IMG])))
print(u.groupby("subject")[[LAB, SCAR]].mean())
