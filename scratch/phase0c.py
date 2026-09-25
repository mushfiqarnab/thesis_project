import pandas as pd, numpy as np, json
from pathlib import Path

CSV        = 'data/csv/multimodal_10k.csv'
SPLIT_PATH = 'outputs/subject_disjoint_split.json' 

df = pd.read_csv(CSV)
assert 'subject' in df.columns, 'No subject column -- check CSV schema'
assert 'threat'  in df.columns
assert 'scar'    in df.columns

subjects = sorted(df['subject'].unique())
print(f'Subjects ({len(subjects)}): {subjects}')

stats = {}
for s in subjects:
    sub = df[df['subject'] == s]
    stats[s] = {
        'n': len(sub),
        'threat_rate': float(sub['threat'].mean()),
        'scar_rate':   float(sub['scar'].mean()),
        'both_classes': len(sub['threat'].unique()) == 2,
    }
    print(f'  {s}: n={stats[s]["n"]}  threat={stats[s]["threat_rate"]:.3f}  scar={stats[s]["scar_rate"]:.3f}  both_classes={stats[s]["both_classes"]}')

eligible = [s for s in subjects if stats[s]['both_classes']]
print(f'Eligible subjects (both threat classes): {eligible}')
if len(eligible) < 4:
    print('WARNING: fewer than 4 eligible subjects. Using all eligible for val.')
    val_subjects = eligible[:2]
else:
    by_rate = sorted(eligible, key=lambda s: stats[s]['threat_rate'])
    step    = len(by_rate) // 3
    val_subjects = [by_rate[0], by_rate[step], by_rate[-1]]

train_subjects = [s for s in subjects if s not in val_subjects]
print()
print(f'Val   subjects ({len(val_subjects)}): {val_subjects}')
print(f'Train subjects ({len(train_subjects)}): {train_subjects}')

train_idx = df[df['subject'].isin(train_subjects)].index.tolist()
val_idx   = df[df['subject'].isin(val_subjects)].index.tolist()

val_df = df.loc[val_idx]
assert len(val_df['threat'].unique()) == 2, 'Val set missing a threat class -- adjust selection'
assert len(val_df['scar'].unique()) == 2,   'Val set missing a scar condition -- adjust selection'
print()
print(f'Val rows: {len(val_idx)}   Val threat rate: {val_df["threat"].mean():.4f}   Val scar rate: {val_df["scar"].mean():.4f}')
print(f'Train rows: {len(train_idx)}')
print()
print('VERIFIED: val set contains both threat classes and both scar conditions.')

Path(SPLIT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(SPLIT_PATH, 'w') as f:
    json.dump({'train_idx': train_idx, 'val_idx': val_idx,
               'train_subjects': train_subjects, 'val_subjects': val_subjects}, f)
print(f'Saved -> {SPLIT_PATH}')
print('Subject-disjoint split written to the exact path train_cgf_fair.py will load.')
