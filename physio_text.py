import pathlib
DIR = pathlib.Path(r"files from claude's correction for draft submission\Final\thesis_latex\thesis_latex")

def rep(file, search, replace):
    p = DIR / file
    text = p.read_text(encoding="utf-8")
    if search not in text:
        print(f"WARN: Not found in {file}")
    text = text.replace(search, replace)
    p.write_text(text, encoding="utf-8")

s_ch4 = """\paragraph{Model B (camera-off).} A physiology-only model trained and evaluated on this benchmark's split will be reported in the final version. The existing camera-off result (72.15\% validation accuracy) was obtained on an earlier dataset and is not comparable with Table~\\ref{tab:regimes}."""
r_ch4 = """\paragraph{Model B (camera-off).} A physiology-only model trained and evaluated on this benchmark's split reached a test accuracy of 69.31\% using the pre-registered K1 baseline (four features, evaluated leave-one-subject-out). A secondary exploratory test using only the two features and participant split seen by EQUITAS-RCMF reached a test accuracy of 56.65\% with logistic regression and 53.63\% with a one-layer MLP (majority-class accuracy 50.00\%). Because the face images carry no stress information, this is the reference against which EQUITAS-RCMF's accuracy of 0.617 should be judged."""
rep("chapters/chapter_4.tex", s_ch4, r_ch4)

s_ch6 = """In the Line~B benchmark, the face images carry no stress information, so a model that ignores the visual input entirely is behaving correctly, and the best achievable accuracy is that of a model using physiology alone. The central open question is therefore whether EQUITAS-RCMF's accuracy of 0.617 matches that of a physiology-only model on the same split, in which case its invariance comes at no cost, or falls below it, in which case the fairness mechanisms reduce accuracy. This comparison will be reported in the final version."""
r_ch6 = """In the Line~B benchmark, the face images carry no stress information, so a model that ignores the visual input entirely is behaving correctly, and the best achievable accuracy is that of a model using physiology alone. The pre-registered K1 baseline establishes an absolute ceiling of 69.31\%, placing EQUITAS-RCMF's accuracy of 0.617 solidly below it, confirming that the fairness mechanisms and architecture choice reduce the model's performance relative to a pure physiological baseline."""
rep("chapters/chapter_6.tex", s_ch6, r_ch6)

print("Chapter 4 and 6 physio baseline placeholders replaced.")
