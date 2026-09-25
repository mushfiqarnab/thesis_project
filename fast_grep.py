import os
import sys

target = '3.566283'
exclude_dirs = {'.venv_worldclass', '.git', 'outputs', 'data', '__pycache__', 'scratch'}

print('### COMMAND 3: Constant Provenance Audit')
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        if file.endswith('.py') or file.endswith('.md'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if target in line:
                            print(f'{path}:{i+1}: {line.strip()}')
            except Exception:
                pass
