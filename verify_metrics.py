import sys
sys.path.insert(0, 'src')
from pathlib import Path
import json

print('[FINAL PROJECT METRICS VERIFICATION]')
print()

# Count files
src_files = list(Path('src').glob('*.py'))
print('[Source Files Count]')
print('  Total in src/:', len(src_files))
print('  Expected: 20')
assert len(src_files) == 20, f'Expected 20 files, got {len(src_files)}'
print('  ✓ Match')
print()

# Code size
total_size = 0
for f in src_files:
    total_size += f.stat().st_size

print('[Code Size]')
print(f'  Total: {total_size:,} bytes (~{total_size/1024:.0f} KB)')
print('  Expected: ~180 KB')
assert 170000 < total_size < 190000
print('  ✓ Match')
print()

# Data files
csv_files = list(Path('data/csv').glob('*.csv'))
print('[Data Files]')
print(f'  CSV files: {len(csv_files)}')
for csv in sorted(csv_files):
    import pandas as pd
    df = pd.read_csv(csv)
    print(f'    - {csv.name}: {len(df)} rows')
print()

# Parameters
from models import MultimodalThreatModel, count_trainable_params
model_concat = MultimodalThreatModel(phys_dim=2, vision_backbone='mobilenet_v3_small', fusion='concat')
model_cgf = MultimodalThreatModel(phys_dim=2, vision_backbone='mobilenet_v3_small', fusion='cgf')

print('[Model Parameters]')
concat_params = count_trainable_params(model_concat)
cgf_params = count_trainable_params(model_cgf)
print(f'  CONCAT: {concat_params:,}')
print(f'  CGF:    {cgf_params:,}')
assert 1010000 < concat_params < 1020000
assert 1160000 < cgf_params < 1170000
print('  ✓ Match')
print()

print('[CONCLUSION]')
print('✓ All project metrics verified and accurate')
