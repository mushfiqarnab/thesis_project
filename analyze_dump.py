import json
import re

dump_path = r'C:\Users\USERAS\.gemini\antigravity-cli\brain\fd1f83c5-6e8d-46f8-9784-03723c1575e8\scratch\repo_dump.txt'
with open(dump_path, 'r', encoding='utf-8') as f:
    content = f.read()

# find all file names
files = re.findall(r'FILE: (.*)', content)
print("Files found in repo:")
for file in files:
    print(f" - {file}")

