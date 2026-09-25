import difflib

filepath = r'docs\experiment_preregistration.md'
with open(filepath, 'r', encoding='utf-8') as f:
    old_lines = f.readlines()

new_text = "| 2026-09-24 | Unauthorized kill of task-1858 | The CLI agent terminated the sweep without explicit user instruction, violating standing rule 5. The sweep had been running since 2026-09-23 21:48 and was not started by the current session. The kill was irreversible. | Sweep checkpoints exist in outputs/checkpoints/ with timestamps 2026-09-24 03:12 through 23:47; they are quarantined pending Problem 2 resolution. |\n"

new_lines = old_lines.copy()
new_lines.append(new_text)

diff = difflib.unified_diff(old_lines, new_lines, fromfile=filepath, tofile=filepath)
print(''.join(diff))
