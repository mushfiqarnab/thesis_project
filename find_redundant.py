import os
import glob

print('--- Redundant & Legacy Files Analysis ---')

# 1. Backup and Legacy Source Code
backup_dirs = glob.glob('src/**/*.backup', recursive=True)
print('\n### 1. Backup & Legacy Code Directories (Safe to Delete)')
for d in backup_dirs:
    print(f'- {d} (Contains redundant pre-refactor code)')

legacy_py = ['src/models/train_empirical.py', 'src/models/train_fair_empirical.py', 'src/train_cgrn_phase4.py']
print('\n### 2. Orphaned / Deprecated Training Scripts')
for f in legacy_py:
    if os.path.exists(f):
        print(f'- {f} (Replaced by train_equitas_rcmf.py and ultimate_sweep.py)')

# 2. Redundant Markdown Documents
docs = glob.glob('*.md') + glob.glob('docs/*.md')
essential_docs = ['README.md', 'docs\\FULL_THESIS_PAPER.md']
print('\n### 3. Redundant / Intermediate Planning Documents')
for doc in docs:
    if doc not in essential_docs and 'FULL_THESIS_PAPER' not in doc:
        print(f'- {doc}')

# 3. Intermediate Temp Files
temp_csvs = glob.glob('outputs/temp_eval_rcmf_*.csv')
print('\n### 4. Temporary Execution Artifacts')
for t in temp_csvs:
    print(f'- {t}')
    
if not temp_csvs:
    print('- No leaked temp CSV files found currently.')

