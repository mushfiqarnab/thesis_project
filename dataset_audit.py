import pandas as pd

df = pd.read_csv('data/csv/multimodal_10k.csv')
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print()
print('HRV stats:')
print(df['hrv'].describe())
print()
print('GSR stats:')
print(df['gsr'].describe())
print()
numeric_cols = df.select_dtypes('number').columns.tolist()
print('Correlations with threat:')
for col in numeric_cols:
    if col != 'threat':
        r = df[col].corr(df['threat'])
        print(f'  {col:<20} r = {r:.4f}')
print()
print('Correlations with scar:')
for col in numeric_cols:
    if col != 'scar':
        r = df[col].corr(df['scar'])
        print(f'  {col:<20} r = {r:.4f}')
