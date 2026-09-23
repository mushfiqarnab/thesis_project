import subprocess
import os

preprocessor = r'C:\Users\USERAS\.gemini\antigravity-cli\brain\24182451-798e-4bbb-9ff2-cce21e69f586\scratch\preprocess_video_mediapipe.py'
evaluator = r'C:\Users\USERAS\.gemini\antigravity-cli\brain\24182451-798e-4bbb-9ff2-cce21e69f586\scratch\run_full_pos_pipeline.py'

for alpha in ['1.0', '0.5']:
    print(f'\n======================================')
    print(f'STARTING PASS WITH ALPHA = {alpha}')
    print(f'======================================')
    
    for s in ['s1', 's2', 's3', 's4']:
        for t in ['1', '2', '3']:
            print(f'Running extraction for {s} T{t} (alpha={alpha})...')
            subprocess.run(['python', preprocessor, s, t, alpha])
    
    print(f'Extraction complete for alpha={alpha}. Running POS Evaluation...')
    subprocess.run(['python', evaluator])
