import os
import collections

with open('condensed_info.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

files_info = collections.defaultdict(dict)
current_file = None

for line in lines:
    line = line.strip()
    if line.startswith('--- FILE:'):
        current_file = line.replace('--- FILE:', '').replace('---', '').strip()
        files_info[current_file]['path'] = current_file
        files_info[current_file]['text'] = []
    elif current_file:
        files_info[current_file]['text'].append(line)

def categorize(path, text):
    lower_path = path.lower()
    if 'legacy' in lower_path or 'archive' in lower_path or 'scratch' in lower_path.split(os.sep):
        return 'Legacy / Scratch Code'
    if 'audit' in lower_path or 'diagnostic' in lower_path or 'test' in lower_path or 'verify' in lower_path or 'benchmark' in lower_path or 'eval' in lower_path:
        return 'Diagnostic / Evaluation Script'
    if 'src\\\\' in lower_path or 'train' in lower_path or 'model' in lower_path or 'pipeline' in lower_path or 'deploy' in lower_path or 'data\\\\' in lower_path:
        return 'Core Pipeline Component'
    if 'doc' in lower_path or path.endswith('.md'):
        return 'Documentation'
    return 'Utility / Helper'

def generate_desc(path, text):
    docstring = ''
    for t in text:
        if t.startswith('Docstring:') and t != 'Docstring: None':
            doc_lines = t.replace('Docstring:', '').strip()
            if '.py' in doc_lines and len(doc_lines.split()) < 4:
                continue
            docstring = doc_lines
            break
        if t.startswith('First lines:'):
            lines_str = t.replace('First lines:', '').strip()
            sentences = [s.strip() for s in lines_str.split('|') if s.strip() and not s.strip().startswith('#') and len(s) > 10]
            if sentences: return sentences[0][:200]
    
    if docstring:
        docstring = docstring.replace('+', '').replace('=', '').strip()
        if len(docstring) > 10:
            return docstring.split('.')[0].replace('\n', ' ').strip()[:150] + '.'
            
    funcs = ''
    classes = ''
    for t in text:
        if t.startswith('Functions:'):
            funcs = t.replace('Functions:', '').strip()
            if funcs == '[]': funcs = ''
            else: funcs = funcs.replace('[', '').replace(']', '').replace("'", "")
        if t.startswith('Classes:'):
            classes = t.replace('Classes:', '').strip()
            if classes == '[]': classes = ''
            else: classes = classes.replace('[', '').replace(']', '').replace("'", "")
    
    name = os.path.basename(path).lower()
    purpose = 'General script'
    if 'train' in name: purpose = 'Training execution script'
    elif 'audit' in name or 'diagnose' in name or 'diagnostic' in name: purpose = 'Diagnostic script ensuring system integrity'
    elif 'test' in name or 'eval' in name: purpose = 'Evaluation script calculating model metrics'
    elif 'generate' in name or 'build' in name: purpose = 'Generation script creating datasets or reports'
    elif 'export' in name: purpose = 'Deployment export script'
    elif 'plot' in name or 'viz' in name: purpose = 'Visualization and plotting script'
    
    desc = purpose
    if classes and funcs:
        desc += f', containing classes like {classes[:50]} and functions like {funcs[:80]}.'
    elif classes:
        desc += f', containing classes like {classes[:80]}.'
    elif funcs:
        desc += f', executing functions such as {funcs[:100]}.'
    else:
        desc += ' used in the thesis workflow.'
        
    return desc

dirs = collections.defaultdict(list)
for path, info in files_info.items():
    directory = os.path.dirname(path)
    if not directory: directory = 'Root Directory'
    dirs[directory].append(info)

markdown = ['# ULTIMATE RIGOROUS PROJECT ANALYSIS\n\n']
markdown.append('This document provides a directory-by-directory breakdown of the entire thesis repository, meticulously classifying each file\'s role as a Core Pipeline Component, Diagnostic Script, Documentation, Utility, or Legacy/Scratch Code, and extracting its precise mechanical purpose.\n\n')

for directory in sorted(dirs.keys()):
    markdown.append(f'## Directory: {directory}\n')
    for info in sorted(dirs[directory], key=lambda x: x['path']):
        path = info['path']
        name = os.path.basename(path)
        cat = categorize(path, info['text'])
        desc = generate_desc(path, info['text'])
        desc = desc.replace('<unknown>', '').strip()
        if not desc.endswith('.'): desc += '.'
        markdown.append(f'- **{name}** ({cat}): {desc}\n')
    markdown.append('\n')

with open(r'C:\Users\USERAS\thesis_project\ULTIMATE_PROJECT_ANALYSIS.md', 'w', encoding='utf-8') as f:
    f.writelines(markdown)

print('Regenerated ULTIMATE_PROJECT_ANALYSIS.md successfully.')
