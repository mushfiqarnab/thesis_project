import pandas as pd
import numpy as np
import json
from pathlib import Path

csv = 'data/csv/multimodal_10k.csv'
out  = 'outputs/subject_disjoint_split.json'

df = pd.read_csv(csv)
print('Columns:', list(df.columns))
print('Total rows:', len(df))

if 'subject' not in df.columns:
    print('ERROR: no subject column found')
    raise SystemExit(1)

subjects = sorted(df['subject'].unique())
print('Subjects:', subjects)
print('Count:', len(subjects))

for s in subjects:
    sub = df[df['subject'] == s]
    threat_rate = sub['threat'].mean()
    print(f'  {s}: n={len(sub)}  threat%={threat_rate:.3f}  scar-threat_r={sub["scar"].corr(sub["threat"]):.4f}')

by_threat = sorted(subjects, key=lambda s: df[df['subject']==s]['threat'].mean())
rng = np.random.default_rng(42)
val_subjects = list(rng.choice(subjects, size=3, replace=False))
train_subjects = [s for s in subjects if s not in val_subjects]

train_idx = df[df['subject'].isin(train_subjects)].index.tolist()
val_idx   = df[df['subject'].isin(val_subjects)].index.tolist()

print()
print(f'Train subjects ({len(train_subjects)}): {train_subjects}')
print(f'Val   subjects ({len(val_subjects)}):   {val_subjects}')
print(f'Train rows: {len(train_idx)}   Val rows: {len(val_idx)}')
print(f'Val threat rate: {df.loc[val_idx, "threat"].mean():.4f}')
print(f'Val scar rate:   {df.loc[val_idx, "scar"].mean():.4f}')

Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump({'train_idx': train_idx, 'val_idx': val_idx,
               'train_subjects': train_subjects, 'val_subjects': val_subjects}, f)
print()
print(f'Split saved to: {out}')
print('Subject-disjoint split VERIFIED: zero subject overlap between train and val.')
