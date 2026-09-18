import pandas as pd
import os

csv_path = 'data/csv/multimodal_10k.csv'
if not os.path.exists(csv_path):
    print(f'ERROR: {csv_path} not found on disk.')
else:
    df = pd.read_csv(csv_path)
    print('===================================================')
    print('         MANIFOLD CORRELATION PROOF (LEGACY)       ')
    print('===================================================')
    
    # Calculate exact Pearson correlation
    pearson_r = df['scar'].corr(df['threat'])
    print(f'Pearson Correlation (r) between Scar and Threat: {pearson_r:.4f}')
    
    if pearson_r > 0.85:
        print('DIAGNOSIS: CRITICAL SEVERE CORRELATION DETECTED.')
        print('The dataset inherently binds the demographic/visual proxy to the label.')
    
    print('\n--- Joint Distribution Cross-Tabulation ---')
    crosstab = pd.crosstab(index=df['scar'], columns=df['threat'], 
                           margins=True, margins_name='Total')
    crosstab.index.name = 'Scar Feature (0/1)'
    crosstab.columns.name = 'Threat Label (0/1)'
    print(crosstab)
    print('===================================================')
