import os
import glob

def dump_files():
    root = r'C:\Users\USERAS\thesis_project'
    dump_path = r'C:\Users\USERAS\.gemini\antigravity-cli\brain\fd1f83c5-6e8d-46f8-9784-03723c1575e8\scratch\repo_dump.txt'
    
    # Ensure scratch dir exists
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)

    patterns = [
        'src/**/*.py',
        'docs/**/*.md',
        'outputs/reports/**/*.json',
        'outputs/*.csv',
        '*.md',
        'requirements.txt',
        'pyproject.toml',
        'setup.py'
    ]
    
    with open(dump_path, 'w', encoding='utf-8') as outfile:
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(root, pattern), recursive=True):
                # skip the thesis paper we just wrote to avoid recursion/duplication
                if 'FULL_THESIS_PAPER.md' in filepath:
                    continue
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(f'\n\n{"="*80}\n')
                        outfile.write(f'FILE: {filepath}\n')
                        outfile.write(f'{"="*80}\n')
                        outfile.write(content)
                except Exception as e:
                    outfile.write(f'\nError reading {filepath}: {e}\n')
    print(f'Dump complete: {dump_path}')
dump_files()
