import re
with open('docs/FULL_THESIS_PAPER.md', 'r', encoding='latin-1') as f:
    text = f.read()

text = re.sub(r'\(\ = 0\.0001\$, CHROM\nestimator\)', '( = 0.0314$ (POS, primary pre-registered test), corroborated by CHROM ( = 0.0001$, exploratory) and PBV ( = 0.0103$, exploratory))', text)
text = text.replace('CHROM  = 0.0001$', 'POS  = 0.0314$ (primary, pre-registered); CHROM  = 0.0001$ and PBV  = 0.0103$ (exploratory corroboration)')
text = text.replace('\n  *(Hash verified on 2026-09-23 via PowerShell Get-FileHash against the physical file at data/publishable_scar_production/multimodal_publishable.csv)*', '')

a1_sha = '- **SHA-256:** 2035777f957fa4c5cc830989f4fe40fb9190589f17c567cb29a3ef0e6b2acae2'
a1_sha_new = '- **SHA-256:** 2035777f957fa4c5cc830989f4fe40fb9190589f17c567cb29a3ef0e6b2acae2\n  *(Hash verified on 2026-09-23 via PowerShell Get-FileHash against the physical file at data/publishable_scar_production/multimodal_publishable.csv)*'
text = text.replace(a1_sha, a1_sha_new)

with open('docs/FULL_THESIS_PAPER.md', 'w', encoding='latin-1') as f:
    f.write(text)