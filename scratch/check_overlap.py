import json
import pandas as pd

u = pd.read_csv('data/csv/multimodal_10k_unbiased.csv')
split = json.load(open('data/csv/multimodal_10k_strict_split_seed42.json'))

tr_subs = set(u.iloc[split['train_idx']]['subject'])
val_subs = set(u.iloc[split['val_idx']]['subject'])

print('Unique subjects:', u['subject'].nunique())
print('Max rows/subject:', u['subject'].value_counts().max())
print('Train/Val overlap:', len(tr_subs.intersection(val_subs)))
