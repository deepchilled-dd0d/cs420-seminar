# Presentation Q&A: LoRA and O-LoRA

Likely audience questions with prepared answers, based on [report.tex](../report/report.tex) and its sections, [presentation.tex](presentation.tex), and the companion notes in [presentation_explained.md](presentation_explained.md).

## How to use this file

- **Short answer** is what to say first: two or three sentences, about 20 seconds. Stop there unless the asker wants more.
- **If pressed** gives the detail, numbers or derivation for a follow-up.
- **Slide** is the slide to ask the driver to show. **Answers** is the default speaker, following the talk: Quan (slides 1–5), Tinh (6–9), Lac (10–14).
- **Stay within our evidence.** We ran one LoRA and one full fine-tuning run; we did not run O-LoRA, sweep ranks or measure latency. If a question goes beyond that, say so plainly ("we did not test that") and give the expected answer from the papers, labeled as such.

### Quick reference numbers

| Quantity | Value |
|---|---|
| One $1024\times1024$ matrix, rank 4 | 8,192 vs. 1,048,576 trainable (128×, 99.21875% fewer) |
| Trainable share in our LoRA run | 0.8944% (147,456 adapter + 593,668 classifier = 741,124 of ≈ 83 M) |
| Test accuracy | LoRA 93.76%, full FT 95.03% (gap 1.26 pp; macro F1 gap 1.27 pp) |
| Test errors (of 7,600) | LoRA 474, full FT ≈ 378 |
| Training wall time | 47.31 vs. 66.38 min (≈ 29% faster) |
| Peak allocated GPU memory | 1.099 vs. 2.138 GiB (≈ 49% less) |
| Main confusion | Business→Sci/Tech 139, Sci/Tech→Business 108 (52% of errors) |
| LoRA config | $r=8$, $\alpha=16$, dropout 0.05, query and value, lr $2\times10^{-4}$ (full FT $2\times10^{-5}$) |

---

## 1. Motivation and fundamentals

### Q1. Why should a low-rank update be enough to adapt a pretrained model?

**Short answer:** The pretrained model already holds useful representations, so a new task usually needs changes in only a few directions. LoRA bets that the needed weight change has low rank. That bet is motivated by low intrinsic dimension, but it is a hypothesis, not a guarantee.

**If pressed:** Aghajanyan et al. reached 90% of full fine-tuning performance on MRPC by optimizing only about 200 numbers through a random projection, and larger pretrained models needed fewer. Hu et al. report that very small ranks match full fine-tuning on many benchmarks. It can fail when a task needs broad changes, such as a language or domain far from pretraining.

*Slide 4–5 · Answers: Quan*

### Q2. Doesn't low intrinsic dimension already prove that the update is low rank?

**Short answer:** No. Intrinsic dimension is about the whole parameter vector; rank is about each individual matrix. One does not imply the other, which is why the slide says it "motivates, but does not prove".

**If pressed:** In the random-subspace experiment, the update $P\theta^{(d)}$ gives each weight matrix a random mixture of directions, so each matrix's change is typically *full* rank even though the overall update lives in a tiny subspace. Low rank per matrix is a separate assumption that LoRA adds and then tests by whether it works.

*Slide 5 · Answers: Quan*

### Q3. How does LoRA differ from linear probing, full fine-tuning and adapters?

**Short answer:** Linear probing trains only a new head, so internal features cannot change. Full fine-tuning updates everything, which is costly per task. LoRA freezes the pretrained weights but adds trainable low-rank factors inside selected layers, so internal representations can adapt with few trainable parameters.

**If pressed:** Unlike adapters, LoRA's update is linear, so it can be merged into $W_0$ after training with no inference cost; adapters contain a nonlinearity and always add latency. Prompt and prefix tuning leave weights untouched and use up sequence length. Note that our LoRA run also trains the classifier head, as every method must.

*Slide 3 · Answers: Quan*

---

## 2. LoRA formulation and parameter savings

### Q4. What do the rank and the scaling factor control?

**Short answer:** In $h=W_0x+\frac{\alpha}{r}ABx$, the rank $r$ limits how many independent directions the update can change: its capacity. The factor $\alpha/r$ keeps the update's size stable when $r$ changes, so the learning rate does not have to be retuned for each rank.

**If pressed:** A larger $r$ sums more rank-1 terms, so without the division the update would grow just from changing $r$. We used $\alpha=16$, $r=8$, i.e. a scale of 2. Larger ranks add capacity and cost; we did not test their effect on accuracy.

*Slide 5 · Answers: Quan*

### Q5. Why initialize one factor to zero, and why not both?

**Short answer:** With $B=0$, the product $AB$ is zero, so training starts from exactly the pretrained model and the first steps cannot damage it. Both cannot be zero, because then neither factor would ever receive a gradient.

**If pressed:** $\partial\mathcal L/\partial A\propto(\cdot)(Bx)^\top$ and $\partial\mathcal L/\partial B\propto A^\top(\cdot)$. With $A$ random and $B=0$, $B$ gets a gradient at step 1, and $A$ starts moving once $B\neq0$. Which factor is zero is a convention: the slides follow the report (A Gaussian, B zero); the LoRA paper swaps the letter names.

*Slide 5 · Answers: Quan*

### Q6. How do you get the 99.22% reduction?

**Short answer:** A $d\times k$ matrix has $dk$ weights; its LoRA factors have $r(d+k)$. For a $1024\times1024$ matrix at rank 4, that is 8,192 instead of 1,048,576: 128 times fewer, a 99.21875% reduction.

**If pressed:** For a square $n\times n$ matrix the ratio is $n/(2r)$, so the saving grows with matrix size: the same rank on a GPT-3-sized matrix ($n=12{,}288$) gives about 1,536×. LoRA stops saving parameters once $r$ reaches $dk/(d+k)$, which is 512 here. This example covers one matrix, not a whole model.

*Slide 8 · Answers: Tinh*

### Q7. If each matrix is 48× smaller, why is 0.8944% of the model trained rather than much less?

**Short answer:** Because most of what we train is the classifier head, not the adapters. The head is new and must be trained by every method.

**If pressed:** The twelve rank-8 query and value adapters total 147,456 parameters; the classifier head has 593,668. Together that is 741,124 of about 83 million, or 0.8944%. The adapters alone are under 0.2% of the model, and the head is about 80% of the trained parameters.

*Slides 8, 12 · Answers: Tinh*

### Q8. How do you choose the rank, and why adapt only query and value?

**Short answer:** Rank is chosen empirically, typically 4 to 64: start small and increase until validation quality stops improving. We used rank 8 on query and value, following the LoRA paper's finding that these give most of the benefit for a fixed budget.

**If pressed:** We did not sweep ranks or target modules, so we cannot say whether rank 16 or adapting the MLP layers would close our 1.26-point gap. That is one of our listed next steps.

*Slides 12, 14 · Answers: Lac*

---

## 3. Training cost, memory and deployment

### Q9. Why does a 100×+ parameter reduction give only about half the GPU memory?

**Short answer:** LoRA removes gradients and optimizer states for the frozen weights, but it still stores those weights and still needs activations and temporary tensors. In our runs, peak allocated memory was 1.099 GiB for LoRA versus 2.138 GiB for full fine-tuning.

**If pressed:** With Adam, full fine-tuning stores about four numbers per weight (the weight, its gradient, two optimizer states); LoRA keeps one. So on weight-related memory the saving can never exceed about 4×; the slide's example gives 3.9×. Activations are the same for both methods and pull the real ratio lower. The LoRA paper reports the same pattern on GPT-3: about 10,000× fewer trainable parameters but only about 3× less memory.

*Slides 9, 13 · Answers: Tinh*

### Q10. Why is LoRA training only about 29% faster?

**Short answer:** Freezing $W_0$ does not skip it. The forward pass still multiplies by $W_0$, and the backward pass still sends gradients through it to earlier layers. LoRA only skips computing the gradient *for* $W_0$ and the optimizer update.

**If pressed:** 47.31 vs. 66.38 minutes is a 19-minute, ≈ 29% saving. This was one run each on one GPU, so the exact figure is indicative only.

*Slides 6, 13 · Answers: Tinh*

### Q11. Does LoRA add inference latency? Should we merge the adapter?

**Short answer:** If merged, no: $W_{\rm merged}=W_0+\frac{\alpha}{r}AB$ is a single matrix of the original size, so the layer costs exactly what it did before. Keeping the adapter separate adds a small extra branch but makes task switching easy.

**If pressed:** Merging works because the update is linear: $W_0x+sABx=(W_0+sAB)x$. To switch tasks after merging, subtract the update again or keep a copy of $W_0$. We did not measure merged versus unmerged latency; our ≈ 30-second evaluation runtimes are not a latency test.

*Slide 7 · Answers: Tinh*

### Q12. What is the benefit of keeping many small adapters?

**Short answer:** Storage and flexibility: one shared base model plus a small adapter per task, instead of a full model copy per task. Each adapter only works with the exact base model it was trained on.

**If pressed:** In the LoRA paper, the GPT-3 per-task checkpoint shrank from about 350 GB to about 35 MB. In our run, the adapter weights are about 0.6 MB in FP32 (plus about 2.4 MB for the classifier head), versus about 330 MB for the full model.

*Slide 7 · Answers: Tinh*

---

## 4. O-LoRA and continual learning

### Q13. How does O-LoRA try to reduce catastrophic forgetting?

**Short answer:** It learns a separate LoRA for each new task while freezing the base model and all earlier adapters, and it adds a penalty $\lambda\sum_{i<t}\|A_i^\top A_t\|_F^2$ that discourages the new update from using the same directions as earlier ones. It does this without replaying old data.

**If pressed:** The update $A_tB_tx$ always lies in the column space of $A_t$, so those columns are the output directions task $t$ changes. If they are orthogonal to earlier tasks' directions, the new task should interfere less with what the earlier ones learned.

*Slides 10–11 · Answers: Lac*

### Q14. Does orthogonality guarantee no forgetting? What does O-LoRA cost?

**Short answer:** No. A finite penalty only encourages approximate orthogonality, and even orthogonal directions in one layer do not guarantee unchanged predictions, since later layers mix them. Storage also grows with the number of tasks.

**If pressed:** The accumulated updates can be merged for inference, so no task ID is needed. We reviewed O-LoRA from the paper only and did not evaluate it.

*Slide 11 · Answers: Lac*

### Q15. Why does the penalty involve only $A$ and not $B$?

**Short answer:** Because $A$ decides *which* output directions a task can change; $B$ only decides *how much* of each direction to use for a given input. Constraining $A$ is enough to separate the tasks' update subspaces.

**If pressed:** $sA_tB_tx=A_t(sB_tx)$ is a linear combination of $A_t$'s columns for any input $x$.

*Slide 10 · Answers: Lac*

### Q16. Why does $A_i^\top A_t=0$ mean the subspaces are orthogonal, and why the Frobenius norm?

**Short answer:** Entry $(p,q)$ of $A_i^\top A_t$ is the inner product of column $p$ of $A_i$ with column $q$ of $A_t$. All entries zero means every pair of columns is orthogonal, and so is everything they span. The squared Frobenius norm adds up all those overlaps into one smooth number that is zero only at exact orthogonality, which makes it usable as a loss.

**If pressed:** $(A_ia)^\top(A_tb)=a^\top(A_i^\top A_t)b$ for any vectors in the two column spaces, so checking $r^2$ column pairs replaces checking infinitely many vectors.

*Slide 10 · Answers: Lac*

### Q17. How many tasks can O-LoRA handle before the directions run out?

**Short answer:** At most $\lfloor d/r\rfloor$ mutually orthogonal rank-$r$ subspaces fit in a $d$-dimensional space: 96 for $d=768$, $r=8$. Beyond that, overlap is unavoidable and the penalty cannot reach zero.

**If pressed:** In practice the constraint is soft anyway, so performance degrades gradually rather than failing at a hard limit. We did not test this.

*Slide 10 · Answers: Lac*

### Q18. Why not just keep one separate LoRA per task?

**Short answer:** That avoids interference completely, but at test time you must know which task each input belongs to in order to pick the right adapter. O-LoRA merges all updates into one model, so it needs no task label.

**If pressed:** Both approaches need no old data and both store one small adapter per task. The difference is task-ID-free inference versus guaranteed isolation.

*Slide 11 · Answers: Lac*

### Q19. How would you choose $\lambda$?

**Short answer:** It is a trade-off: larger $\lambda$ gives more orthogonality but less freedom to fit the new task; too small and the penalty does nothing. It would be tuned on validation data. We did not run O-LoRA, so we have no value of our own to report.

*Slide 11 · Answers: Lac*

---

## 5. Experimental evidence

### Q20. What did the AG News experiment show, and when would you choose LoRA?

**Short answer:** Training 0.8944% of the parameters, LoRA reached 93.76% accuracy versus 95.03% for full fine-tuning, while training about 29% faster with about half the peak memory. LoRA is attractive when those savings justify the quality loss; full fine-tuning suits a stricter quality target when resources allow.

*Slide 13 · Answers: Lac*

### Q21. Is a 1.26-point gap meaningful?

**Short answer:** In error counts it is noticeable: about 378 errors for full fine-tuning versus 474 for LoRA, roughly 25% more mistakes. But with one run per method and different learning rates, we cannot say how much of it is the method and how much is noise.

**If pressed:** The accuracy gap is 1.2631 pp and the macro-F1 gap 1.2661 pp. A fair test would use several seeds per method and report the spread.

*Slides 13–14 · Answers: Lac*

### Q22. Isn't the comparison unfair if the learning rates differ?

**Short answer:** Each learning rate is the standard choice for its method: LoRA's factors start at zero and are few, so they need a larger step; full fine-tuning uses a small step to avoid damaging pretrained features. But it does mean two things differ at once, which is why we list it as a limitation.

**If pressed:** The fix is to tune the learning rate for each method separately on validation data and compare each method at its best setting.

*Slides 12, 14 · Answers: Lac*

### Q23. Why are Business and Sci/Tech confused so often?

**Short answer:** Many articles genuinely belong to both, for example a tech company's earnings or a telecom deal. That one confusion accounts for 247 of the 474 errors, over half.

**If pressed:** It also explains the two weakest per-class numbers: Business has the lowest recall (0.897) because its articles are predicted as Sci/Tech, and Sci/Tech has the lowest precision (0.902) because it receives them. Part of this error may be label ambiguity rather than a weakness of LoRA.

*Slide 13 · Answers: Lac*

### Q24. Why are accuracy and macro F1 almost identical?

**Short answer:** The test set is balanced, 1,900 articles per class, and the model's errors are spread fairly evenly, so the class-averaged F1 ends up very close to accuracy.

*Slide 13 · Answers: Lac*

### Q25. How strong is the comparison overall, and what should come next?

**Short answer:** These are observations from one run per method, with different learning rates, one rank and no variance estimate. The notebook specifies rank 8, but the saved LoRA result files do not record it. Next steps: repeated seeds, separately tuned learning rates, a rank sweep with full metadata, merged versus unmerged latency, and a sequential-task test of O-LoRA.

*Slide 14 · Answers: Lac*

---

## 6. Broader questions

### Q26. When is LoRA the wrong choice?

**Short answer:** When the task needs broad changes in behavior that a low-rank update cannot capture; when the chosen rank cannot reach the quality target; or for small models with plenty of compute, where full fine-tuning is simpler and the savings matter little.

**If pressed:** Our own full fine-tuning needed only about 2.1 GiB of GPU memory, so for a model this size LoRA's memory saving is not decisive.

*Slide 14 · Answers: Lac*

### Q27. Why use DistilRoBERTa and AG News rather than a large language model?

**Short answer:** They fit our compute: both runs finish on one GPU in about an hour, and AG News is a clean, balanced, standard benchmark. The trade-off is that LoRA's advantages grow with model size, so a small model understates them.

*Slide 12 · Answers: Lac*

### Q28. Are there LoRA variants that reduce memory further?

**Short answer:** Yes, but they are outside our study. For example, QLoRA (Dettmers et al., 2023) stores the frozen base model in 4-bit precision, which attacks exactly the "frozen weights still take memory" floor we showed on slide 9.

**If pressed:** We have not evaluated any variants, so we would only point to the idea, not to numbers.

*Slide 9 · Answers: Tinh*
