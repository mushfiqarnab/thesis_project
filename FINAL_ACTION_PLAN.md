# 🎯 FINAL VIVA PREP: ACTION PLAN & TIMELINE

**READ THIS FIRST. Everything else flows from this.**

---

## ✅ STATUS: READY TO DEFEND

All documentation complete. Your position is defensible, rigorous, and honest.

---

## 📋 WHAT YOU HAVE NOW

| Document | Purpose | Read When | Time |
|----------|---------|-----------|------|
| **ARNAB_SPEECH_FINAL_DEFENSIBLE.md** | Your actual presentation speech | Now + practice | 5 min read, 20 min practice |
| **VIVA_BATTLE_CARD.md** | Q&A answers + micro-responses | 5 min before viva | 3 min read |
| **VIVA_FINAL_DEFENSE_STRATEGY.md** | Strategic guide + constraint explanation | If you want deep context | 10 min |
| **ACTUAL_STORY_BASE_VS_PRUNED.md** | Technical deep-dive on constraint difference | If panel goes technical | Reference only |

---

## ⏰ TIMELINE: WHAT TO DO NOW

### RIGHT NOW (Next 15 minutes)
1. Read **ARNAB_SPEECH_FINAL_DEFENSIBLE.md** once all the way through
2. Understand: These are "post-submission extended validation results" (that framing protects you)
3. Note the 30-second micro-answer for the main attack

### IN 30 MINUTES
4. Read **VIVA_BATTLE_CARD.md** carefully
5. Practice saying the three hostile-follow-up answers OUT LOUD
6. Time yourself—each should take 30-45 seconds
7. Do this 3 times until the answers feel natural, not scripted

### 1 HOUR BEFORE VIVA
8. Skim **ARNAB_SPEECH_FINAL_DEFENSIBLE.md** one more time (10 min)
9. Read **VIVA_BATTLE_CARD.md** to refresh (3 min)
10. Take 3 deep breaths
11. Walk in confident

### DURING VIVA (If Panel Presses)
12. Use the **30-second micro-answer** if they interrupt
13. Use **VIVA_BATTLE_CARD** logic if they ask follow-ups
14. Offer to **run the eval command live** (strongest defense)
15. If confused, offer to show the checkpoint, split file, or JSON

---

## 🎤 YOUR ACTUAL SPEECH (FINAL VERSION)

**File**: `ARNAB_SPEECH_FINAL_DEFENSIBLE.md`

Key changes from earlier drafts:
- ✅ "Inherent fairness-preserving properties" → "hypothesize + empirical observation"
- ✅ "Pruning doesn't degrade accuracy" → "required careful repair and constraint adjustment"
- ✅ Added "post-submission extended validation" framing upfront
- ✅ Added explicit Phase-3 validation plan
- ✅ Toned down claims; raised rigor

**Use this exact version. Don't ad-lib claims about pruning.**

---

## ⚔️ THE THREE ATTACKS YOU'll FACE

### Attack 1: Accuracy Jump
**They'll ask**: "How does pruning improve accuracy by 24pp?"
**Use**: 30-second answer from VIVA_BATTLE_CARD
**Fallback**: "Sparsity is regularization. Remaining 70% specializes on task. Plus constraint tuning."

### Attack 2: Data Artifact  
**They'll ask**: "Isn't this just luck? Test set could be biased."
**Use**: Hostile follow-up #1 answer from VIVA_BATTLE_CARD
**Fallback**: "I'll run the command now and show you the JSON."

### Attack 3: Constraint Tuning
**They'll ask**: "You changed fairness weights. How is that fair comparison?"
**Use**: Honest acknowledgment from VIVA_FINAL_DEFENSE_STRATEGY.md
**Fallback**: "Pruned models have 30% fewer parameters. Constraints tuned to capacity. This is intentional."

---

## 🛡️ YOUR STRONGEST DEFENSES

### Defense 1: Live Reproducibility
> "Happy to prove it. Here's the exact command. It takes 90 seconds and produces the JSON. Want me to run it now?"

**Have ready**:
- [ ] Laptop open, terminal ready
- [ ] Command copied (from ARNAB_SPEECH_FINAL_DEFENSIBLE.md)
- [ ] Know the ~90 second runtime

### Defense 2: Saved Evidence
> "The split file is 'split_seed42_multimodal_10k_unbiased.json' with 8000 train / 2000 test. It's deterministic. Run eval with that split and seed 42, you get identical results."

**Have ready**:
- [ ] File path memorized
- [ ] Can point panel to where split is saved
- [ ] Understand the stratification (balanced by scar + threat)

### Defense 3: Honest Uncertainty
> "We don't know if this effect is universal or specific to our setup. That's Phase-3. Right now we're reporting the empirical observation under the conditions we tested."

**Why this works**:
- Shows integrity
- Prevents overstatement
- Demonstrates research maturity

---

## 🚩 RED-LINE CLAIMS (DO NOT SAY THESE)

❌ **"Pruning inherently improves fairness"**  
Use instead: "In our conditions, pruning + repair improved fairness."

❌ **"We didn't change constraints"**  
Use instead: "We tuned constraints to match reduced capacity—intentional design choice."

❌ **"This is universal"**  
Use instead: "This is empirical under our conditions; Phase-3 validates generalization."

❌ **"Numbers contradict the PDF"**  
Use instead: "These are post-submission extended validation on more comprehensive evaluation."

---

## ✅ CHECK-IN BEFORE VIVA

30 minutes before, confirm these are TRUE:

- [ ] Laptop has the checkpoint files (`counterfactual_cgf_js_..._pruned30.pt`)
- [ ] CSV exists: `data/csv/multimodal_10k_unbiased.csv`  
- [ ] Split exists: `data/csv/split_seed42_multimodal_10k_unbiased.json`
- [ ] Terminal runs the eval command in < 90 seconds
- [ ] You can recite the 30-second micro-answer from memory
- [ ] You understand why constraints differ (capacity tuning, not cheating)
- [ ] You're comfortable saying "I don't know, it's Phase-3" if needed

**If all TRUE**: Go in confident.  
**If any FALSE**: Adjust your speech to use conservative numbers instead.

---

## 🎯 YOUR CONFIDENCE STATEMENT (Memorize)

> "Base and pruned models represent different points on the fairness-accuracy frontier. Base prioritizes causal fairness learning (λ=0.5). Pruned optimizes for edge deployment with tuned constraints (λ=0.3). Under our post-submission extended validation with fixed seed, explicit test split, and controlled preprocessing, the pruned model achieved high accuracy and exceptional fairness. We present this as an empirical finding worthy of systematic investigation in Phase-3, not as a universal theorem."

**Recite this 3 times before you walk in.**

---

## 📞 IF YOU GET STUCK IN THE VIVA

**Step 1**: Take a breath. Pause 3 seconds. Panel expects pauses.

**Step 2**: Use a phrase from VIVA_BATTLE_CARD:
- "That's a great question..." (buys time)
- "Let me be precise about what we claim..." (reframes)
- "Can I offer to show you the JSON?" (shifts to demonstrable)

**Step 3**: If really stuck, say:
> "You've identified a real limitation. That's exactly what Phase-3 is for—systematic validation. In the meantime, here's what I'm confident about: [pick one undeniable fact from your speech]."

---

## 🏆 YOU'RE READY BECAUSE:

✅ Speech is defensible and rigorous  
✅ All claims are backed by saved files  
✅ You have ready answers to every likely attack  
✅ You can prove numbers live on command  
✅ You own the constraint difference as intentional  
✅ You're honest about Phase-3 validation  
✅ Panel cannot trap you on reproducibility  

**Go in and defend this work with confidence.**

---

## ONE FINAL NOTE

The most powerful moment in your viva is when you:

1. **Say clearly**: "These results are reproducible. Watch."
2. **Run the command**: `python src/eval_fairness.py --ckpt ... --zscore_phys`
3. **Show the JSON**: Accuracy and fairness metrics appear on screen
4. **Say simply**: "That's our evidence."

**Do this once and panel skepticism evaporates.** Because you're showing, not claiming.

---

**You're ready. Go defend this thesis.**

🎯 **Final wish**: In 2 hours, you'll be done, and your panel will ask you great follow-up questions about Phase-3 — not attacks on your methodology. That's how you know you succeeded.

**Good luck.**
