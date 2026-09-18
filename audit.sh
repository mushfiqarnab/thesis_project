#!/bin/bash
echo '========== PHASE 0A =========='
find . -type f -name '*.py' | sort

echo '========== PHASE 0B =========='
find . -type f \( -name '*.csv' -o -name '*.mp4' -o -name '*.avi' -o -name '*.pth' \) | sort
wc -l scarbench_data/*.csv 2>/dev/null || echo 'SCARBENCH DATA DOES NOT EXIST'

echo '========== PHASE 0C =========='
python -c "
import importlib, sys
packages = ['torch','torchvision','numpy','pandas','sklearn',
            'onnx','onnxruntime','PIL','cv2','diffusers']
for p in packages:
    try:
        m = importlib.import_module(p)
        print(f'OK  {p}: {getattr(m, \"__version__\", \"no version attr\")}')
    except ImportError:
        print(f'MISSING  {p}')
"

echo '========== PHASE 0D =========='
cat src/models/train_production_gw_cd.py 2>/dev/null || echo 'FILE DOES NOT EXIST'

echo '========== PHASE 0E =========='
cat src/data/dataset_builder.py 2>/dev/null || echo 'FILE DOES NOT EXIST'

echo '========== PHASE 0F =========='
find / -maxdepth 4 -name 'BP4D*' -type d 2>/dev/null | head -10
find / -maxdepth 4 -name '*.bp4d' -o -name '*bp4d*' 2>/dev/null | head -10
echo '---'
ls -la data/ 2>/dev/null || echo 'NO DATA DIRECTORY FOUND'
