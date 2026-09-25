import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path

CSV = 'data/csv/multimodal_10k.csv'
df  = pd.read_csv(CSV).dropna(subset=['hrv','gsr','threat','subject'])

print('='*72)
print('PHYSIOLOGICAL DISCRIMINABILITY SCORE (PDS) PER SUBJECT')
print('='*72)
print(f'{"Subject":8s} | {"n":5s} | {"Threat%":7s} | {"HRV_mu":8s} | {"GSR_mu":8s} | {"AUC":6s} | {"PDS":6s}')
print('-'*72)

rows = []
for subj in sorted(df.subject.unique()):
    s = df[df.subject == subj]
    X = s[['hrv','gsr']].values
    y = s['threat'].values.astype(int)
    if len(np.unique(y)) < 2:
        print(f'{subj:8s} | {len(s):5d} | SKIP: only one threat class')
        continue
    mu, sd = X.mean(0), X.std(0)
    sd  = np.where(sd < 1e-9, 1.0, sd)
    Xz  = (X - mu) / sd
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(Xz, y)
    from sklearn.model_selection import cross_val_predict
    p = cross_val_predict(
        LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
        Xz, y, cv=min(5, np.unique(y, return_counts=True)[1].min()),
        method='predict_proba'
    )[:,1]
    auc = roc_auc_score(y, p)
    pds = (auc - 0.5) / 0.5
    rows.append({'subject': subj, 'n': len(s), 'threat_pct': float(y.mean()),
                 'hrv_mean': float(X[:,0].mean()), 'gsr_mean': float(X[:,1].mean()),
                 'auc_cv': float(auc), 'pds': float(pds)})
    print(f'{subj:8s} | {len(s):5d} | {y.mean():7.4f} | {X[:,0].mean():8.5f} | {X[:,1].mean():8.5f} | {auc:6.4f} | {pds:6.4f}')

print('='*72)
rdf = pd.DataFrame(rows)
high = rdf[rdf.pds > 0.30]['subject'].tolist()
low  = rdf[rdf.pds < 0.05]['subject'].tolist()
print(f'High-PDS (PDS>0.30): {high}  -- physiology informative after debiasing')
print(f'Low-PDS  (PDS<0.05): {low}   -- will collapse toward chance after debiasing')
Path('outputs').mkdir(exist_ok=True)
rdf.to_csv('outputs/pds_per_subject.csv', index=False)
print('Saved -> outputs/pds_per_subject.csv')
print()
print('This table is publishable Figure 1.')
