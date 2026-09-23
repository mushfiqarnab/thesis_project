import pandas as pd
from pathlib import Path
import sys

matched_pairs = 0
for s in ['s1', 's2', 's3', 's4']:
    for t in ['1', '2', '3']:
        p1 = Path(f'processed/{s}/T{t}/manifest_alpha1.csv')
        p2 = Path(f'processed/{s}/T{t}/manifest.csv')
        
        if not p1.exists() or not p2.exists():
            continue
            
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        
        if len(df1) != len(df2):
            print(f'Length mismatch in {s} T{t}')
            continue
            
        if (df1.status != df2.status).any():
            print(f'Status mismatch in {s} T{t}')
            print(df1[df1.status != df2.status])
            continue
            
        matched_pairs += 1

assert matched_pairs == 12, f'Expected 12 matched pairs, found {matched_pairs}. Incomplete test.'
print('SUCCESS: All 12 clips match exactly on OK/FILLED/FAILED status between alpha=1.0 and alpha=0.5 passes.')
