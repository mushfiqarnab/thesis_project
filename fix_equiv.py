import pathlib
DIR = pathlib.Path(r"files from claude's correction for draft submission\Final\thesis_latex\thesis_latex")

def rep(file, search, replace):
    p = DIR / file
    text = p.read_text(encoding="utf-8")
    if search not in text:
        print(f"WARN: Not found in {file}")
    text = text.replace(search, replace)
    p.write_text(text, encoding="utf-8")

s1 = """Because a cohort of four participants falls below the pre-registered power threshold, the study follows the pre-registered fallback to estimation only: tests are carried out at the clip level, and the planned equivalence tests are withdrawn."""
r1 = """Because a cohort of four participants falls below the pre-registered power threshold, the study follows the pre-registered fallback to estimation only: tests are carried out at the clip level."""
rep("chapters/chapter_3.tex", s1, r1)

print("Chapter 3 equivalence mention fixed.")
