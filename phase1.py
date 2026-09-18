import os, hashlib, torch

print('===================================================')
print('        EDGE AI CRYPTOGRAPHIC & SYSTEM AUDIT       ')
print('===================================================')
print(f'Active Backend : PyTorch {torch.__version__}')
print(f'CUDA Available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Hardware Accel : {torch.cuda.get_device_name(0)}')
print('---------------------------------------------------')

files = {
    'Fair Checkpoint': 'outputs/fair_model_best.pth',
    'V5 Compressor': 'src/edge/structural_compressor_v5.py',
    'ONNX Export': 'outputs/pacd_net_edge.onnx',
    'Scarbench Train': 'scarbench_data/scarbench_lite_train.csv',
    'Scarbench Test': 'scarbench_data/scarbench_lite_test.csv',
}

for label, path in files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'[OK] {label:<15}: {path} ({size/1024:.1f} KB)')
        if path.endswith('.pth'):
            with open(path, 'rb') as f:
                h = hashlib.sha256(f.read()).hexdigest()[:16]
            print(f'     └─ SHA-256 Hash : {h}')
    else:
        print(f'[--] {label:<15}: MISSING ({path})')
print('===================================================')
