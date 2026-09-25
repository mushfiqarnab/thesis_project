# Chapter 2 Alignment Report (CGF -> EQUITAS-RCMF)

Per Claude's CLI Brief and the checklist in chapter_2_revisions.tex, you must manually update your local chapter_2.tex to remove the obsolete Phase 2 CGF pipeline references. Apply the following corrections:

1. **Kusner et al. relevance:** Change "HRV and GSR" to "BVP and EDA".
2. **Gohumpu et al. relevance (and 2.1.3):** Change "HRV and GSR as an efficient physiological branch" to "BVP and EDA from a wrist-worn device".
3. **Villarejo et al. relevance:** Change "combines GSR with HRV and facial evidence" to "combines EDA with BVP and facial evidence".
4. **FairGRAPE relevance:** Delete the sentences citing CGF Pruned30 DP results (0.0054 vs 0.0112 vs 0.0274). These are from the obsolete model.
5. **Ramesh et al. & Howard et al. relevance:** Remove references to "the compressed CGF model". Describe the EQUITAS-RCMF deployment instead (JIT compilation; INT8 quantization left as future work).
6. **Lian and Celiktutan relevance:** Change "point estimates on a single 80/20 split" to "point estimates from a single evaluation split".
7. **Liu et al. (JTT) relevance:** Delete the clause claiming "the thesis balances these groups directly during training" (EQUITAS-RCMF does not use the old WeightedRandomSampler).
8. **Arjovsky et al. & Cohen et al. relevance:** Update the claim that "the scar and threat groups are balanced during construction". The UBFC-Phys pipeline varies the scar-label correlation ($\rho = 0.85 / 0.50 / 0.15$).
9. **Anthis & Veitch / Arjovsky et al. relevance (and 2.3):** Change sentences calling biased-train/inverted-test evaluation "future work" to "the thesis defines such a protocol (Chapter 3); its results will be reported in the final version."
10. **Global Search:** Replace the word "threat" with "stress", except where discussing threat profiling as the motivating application.
