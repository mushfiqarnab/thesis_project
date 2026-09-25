import os
import re
try:
    from PyPDF2 import PdfReader
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    from PyPDF2 import PdfReader

try:
    import docx
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    import docx

search_terms = ['BP4D', 'LaTo', 'diffusion transformer', 'photorealistic', 'epidermal layer', 'landmark-tokenized', 'landmark tokenized', 'structural geometry']
pattern = re.compile(r'\b(?:' + '|'.join(re.escape(term) for term in search_terms) + r')\b', re.IGNORECASE)

results = []

def search_text(text, filename, context=""):
    for i, para in enumerate(text.split('\n')):
        if pattern.search(para):
            results.append(f"FILE: {filename}\nLOCATION: {context} (Para {i})\nTEXT:\n{para.strip()}\n")

for root, dirs, files in os.walk('.'):
    if any(skip in root for skip in ['.venv', 'node_modules', '.git', '__pycache__']):
        continue
    for file in files:
        path = os.path.join(root, file)
        if file.endswith('.pdf'):
            try:
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        search_text(text, path, f"Page {i+1}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
        elif file.endswith('.docx'):
            try:
                doc = docx.Document(path)
                for i, para in enumerate(doc.paragraphs):
                    text = para.text
                    if pattern.search(text):
                        results.append(f"FILE: {path}\nLOCATION: Paragraph {i+1}\nTEXT:\n{text.strip()}\n")
            except Exception as e:
                print(f"Error reading {path}: {e}")

if results:
    print("\n---\n".join(results))
else:
    print("Zero hits in .pdf and .docx files.")
