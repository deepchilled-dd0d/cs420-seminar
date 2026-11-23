# Slide-by-Slide Companion — LoRA and O-LoRA

A companion to `presentation.tex` (14 slides). It does **not** repeat what the slides show. For each slide, one line says what is on screen; the rest covers what the slide leaves out: derivations, intuition, worked numbers, caveats and links to other slides. Use it to prepare the talk and the Q&A.

Speakers: Quan (slides 1–5), Tinh (6–9), Lac (10–14). Literature citations are in each slide's footer; there is no references or conclusion slide.

**Notation**

| Symbol | Meaning |
|---|---|
| $W_0\in\mathbb R^{d\times k}$ | a pretrained weight matrix (frozen under LoRA) |
| $x\in\mathbb R^{k}$, $h\in\mathbb R^{d}$ | layer input and layer output |
| $A\in\mathbb R^{d\times r}$, $B\in\mathbb R^{r\times k}$, $s=\alpha/r$ | LoRA factors and scale; $\Delta W=sAB$ |
| $D$, $d$, $d_{\rm int}$ | slide 4 only: full parameter dimension, random-subspace dimension, intrinsic dimension |

> **Two different $d$'s.** On slide 4, $d$ is the dimension of a random search subspace. From slide 5 on, $d$ is the number of rows of $W_0$. Slide 5 avoids the clash by writing "low intrinsic dimension" in words.

> **A and B naming.** The deck puts $A$ on the output side ($d\times r$, Gaussian init) and $B$ on the input side ($r\times k$, zero init). The LoRA paper and the PEFT library swap the letters ($\Delta W=BA$, $B=0$). Same mathematics; the slide 5 footer mentions it.

---

## Slide 1 — Title

**On the slide:** title, subtitle, authors with student IDs, course and date.

Nothing to add; the script opens with an audio/screen check and asks for questions in the chat.

---

## Slide 2 — Outline

**On the slide:** the guiding question and a six-item outline.

**Why this question.** LoRA is usually summarized by one number: how many fewer parameters it trains. Practitioners pay for something else: GPU memory, training time and accuracy. The question is built to separate the two:

- **The parameter count** can be derived on paper: per matrix, $dk\to r(d+k)$ (slides 5 and 8).
- **Quality and resources** must be measured, and the measurement has limits (slides 9, 12–14).

**How the outline maps to slides.** Background (3–5) → method (5–7) → worked examples (8–9) → O-LoRA (10–11) → experiment (12–13) → limitations (14). The question is answered aloud at the end of slide 14 (see *Wrap-up*).

---

## Slide 3 — Pretrained model adaptation

**On the slide:** the pretrain-then-adapt figure and a table of five methods with one formula each.

**Two questions that separate the five methods.**

| Method | What is trained? | Inference cost? | Main weakness |
|---|---|---|---|
| Linear probing | new head only | none | features cannot adapt to the task |
| Full fine-tuning | everything | none | one full model copy per task; optimizer state for every weight |
| LoRA | low-rank $A,B$ | none once merged (slide 7) | relies on the low-rank hypothesis (slide 5) |
| Adapter | small bottleneck MLP | always some | the nonlinearity $\sigma$ prevents folding it into $W_0$ |
| Prompt / prefix | extra input or key/value vectors | uses sequence length | can be harder to optimize; less expressive per parameter |

**Reading the formulas.**
- In the linear-probing formula, $x$ is the *model* input; elsewhere it is a layer input.
- Adapter: $W_{\rm down}$ projects to a small dimension, $\sigma$ is a nonlinearity, $W_{\rm up}$ projects back, and "$h+$" is a residual connection, so a zero adapter leaves the layer unchanged.
- Prefix tuning prepends learned vectors $P_K,P_V$ to the keys and values in every attention layer, so every token can attend to them.

**Key point.** LoRA's update is *linear*. That single property is what makes merging (slide 7) possible and separates it from adapters.

---

## Slide 4 — Intrinsic dimension of an objective

**On the slide:** two 3-D panels, the definition $D=d_{\rm int}+s$, and the random-subspace formula $\theta^{(D)}=\theta_0^{(D)}+P\theta^{(d)}$.

**Why the solution set is large.** Modern networks are heavily over-parameterized, so many weight settings fit the data equally well. Moving *along* the solution set $\mathcal S$ changes nothing; only directions *crossing* it matter. Their number is $d_{\rm int}$.

**When a random subspace hits the solution set.** A $d$-dimensional subspace and an $s$-dimensional set in $D$-space generally intersect when $d+s\ge D$, i.e. $d\ge d_{\rm int}$. Panel (b) is the borderline case $d=d_{\rm int}=2$.

**How it is measured in practice.** Increase $d$ until the model reaches a fixed fraction (typically 90%) of full-training performance; that $d$ is reported as the intrinsic dimension. $P$ is random on purpose: if even an unchosen subspace works, the problem genuinely needs few directions.

**What Aghajanyan et al. found.**
- Optimizing about **200** numbers through a random projection brought RoBERTa to 90% of full fine-tuning performance on the MRPC paraphrase task.
- Larger models, and models pretrained longer, have **lower** intrinsic dimension. Pretraining appears to compress what fine-tuning needs to learn, which is exactly the regime where LoRA is used.

---

## Slide 5 — Why a low-rank update, and LoRA formulation

**On the slide:** left, three short motivation bullets and the hypothesis $\operatorname{rank}(\Delta W)\ll\min(d,k)$; right, the LoRA formula, shapes, objective, scaling, initialization and cost.

### Left column: what the bullets leave out

**Low-rank structure (bullet 1).** Oymak et al. show that the Jacobian of a trained network is approximately low rank: the outputs are sensitive to only a few directions in weight space, so learning mostly happens along those directions.

**Why low intrinsic dimension does not prove low rank (bullet 3).** This is the subtle point of the slide.
- Intrinsic dimension is about the *whole* parameter vector, searched through one random global projection $P$.
- Rank is a property of *each matrix* $\Delta W$.
- In the random-subspace experiment, the update $P\theta^{(d)}$ assigns each weight matrix a random mixture of directions, so each $\Delta W$ is typically **full rank**, even though the whole update lives in a tiny subspace.

So the low rank of each $\Delta W$ is a separate claim. LoRA states it as a hypothesis and lets the method's results test it. Hu et al. report that very small ranks (even $r=1$–$8$ on GPT-3's attention matrices) match full fine-tuning on many benchmarks, which supports it empirically.

**From hypothesis to formula.** Any matrix of rank $\le r$ factors as $AB$ with $A\in\mathbb R^{d\times r}$, $B\in\mathbb R^{r\times k}$. If the hypothesis holds, restricting $\Delta W$ to that form loses little, which is the bridge to the right column.

### Right column: why each design choice

**Shapes.** $Bx\in\mathbb R^r$ compresses, $A(Bx)\in\mathbb R^d$ expands, so $AB$ has the shape of $W_0$ but rank at most $r$.

**Objective.** $\Theta$ is the set of all $(A,B)$ pairs, one per adapted matrix. Only $\Theta$ is optimized; $\Phi_0$ never changes.

**Why $\alpha/r$.** A larger $r$ sums more rank-1 terms, so without the scaling the update would grow just from changing $r$, and the learning rate would need retuning each time. With it, one fixes $\alpha$ (we use $\alpha=16$, i.e. $s=2$ at $r=8$) and varies $r$ freely.

**Why $A$ Gaussian and $B=0$.**
- $B=0$ makes $AB=0$, so step 0 is *exactly* the pretrained model; training cannot damage it on step one.
- Both cannot be zero. The gradients are $\partial\mathcal L/\partial A\propto(\cdot)(Bx)^\top$ and $\partial\mathcal L/\partial B\propto A^\top(\cdot)$, so with both zero neither ever moves.
- With $A\ne0$, $B$ gets a gradient at step 1, and $A$ starts moving once $B\ne0$.

**Cost.** $r(d+k)$ instead of $dk$ per matrix; gradients and optimizer states shrink by the same ratio, and the saved per-task checkpoint shrinks too. Slide 8 puts numbers on this.

---

## Slide 6 — LoRA training

**On the slide:** a diagram of the two paths and four bullets: two paths, scaling, only $A,B$ updated, $W_0$ still stored and used.

**What freezing does and does not save.** The backward pass needs two kinds of gradient:
- **Gradients for activations**, which flow to earlier layers through $W_0^\top$ (and $sB^\top A^\top$). These are still computed, because earlier layers' adapters need them.
- **Gradients for weights.** For $W_0$ this would be $\frac{\partial\mathcal L}{\partial h}x^\top$; it is **not** computed. For the factors:
  $$\frac{\partial\mathcal L}{\partial A}=s\,\frac{\partial\mathcal L}{\partial h}(Bx)^\top,\qquad \frac{\partial\mathcal L}{\partial B}=s\,A^\top\frac{\partial\mathcal L}{\partial h}\,x^\top.$$

So LoRA skips the weight gradient and optimizer step for $W_0$, but the forward pass and most of the backward pass remain. **Training gets faster, but not in proportion to the parameter cut**, which is what slide 13 shows (≈ 29% faster).

**Detail not on the slide.** In our experiment, LoRA dropout (0.05) is applied to $x$ on the adapter path only.

---

## Slide 7 — LoRA after training

**On the slide:** two diagrams, one keeping the adapter separate ($W_0x+sABx$) and one merging it ($W_{\rm merged}=W_0+sAB$).

**Why merging is possible.** Linearity: $W_0x+sABx=(W_0+sAB)\,x$. Computing $W_0+sAB$ once gives a matrix with the original shape, so inference is exactly as fast as the original layer. Adapters (slide 3) cannot do this because $\sigma$ is not linear.

**Why keeping adapters separate is attractive: storage.**
- LoRA paper (GPT-3, $r=4$ on query/value): per-task checkpoint ≈ **35 MB instead of ≈ 350 GB**.
- Our run, FP32: LoRA weights $147{,}456\times4$ B ≈ **0.6 MB**, plus the classifier head ≈ 2.4 MB, versus ≈ 330 MB for the full DistilRoBERTa model.

**Practicalities.**
- To switch tasks after merging, subtract $sAB$ (exact up to floating-point rounding) or keep a clean copy of $W_0$.
- A separate adapter is useless without the exact base model it was trained on.
- Unmerged adapters add one small extra product ($Bx$, then $A(\cdot)$) per adapted layer. The cost is small but nonzero; we did not measure it.

---

## Slide 8 — LoRA example: counting trainable parameters

**On the slide:** the full count $1{,}048{,}576$, the LoRA count $2048r$, the $r=4$ example (128×, 99.21875%), the general $512/r$ rule and a table for $r=4,8,16$.

**Where $512/r$ comes from.** For a square $n\times n$ matrix:
$$\frac{dk}{r(d+k)}=\frac{n^2}{2nr}=\frac{n}{2r}.$$

**What follows from it.**
- **The saving grows with matrix size.** GPT-3's matrices have $n=12{,}288$, so the same $r=4$ gives $1{,}536\times$. This is why LoRA matters most for very large models.
- **Break-even.** LoRA saves parameters only while $r<dk/(d+k)$, which is 512 here. Practical ranks are about 4–64.
- **Linear in $r$.** Doubling $r$ doubles the cost and halves the ratio, as the table shows.

**Connecting to our experiment.**

| | Count |
|---|---|
| One $768\times768$ query or value matrix, full | 589,824 |
| Same matrix, LoRA $r=8$: $8(768+768)$ | 12,288 (48× fewer) |
| All 12 adapted matrices (6 layers × query, value) | 147,456 |
| New classifier head ($768\times768+768$, $768\times4+4$) | 593,668 |
| **Trainable total** | **741,124 = 0.8944% of ≈ 83 M** |

So **80% of what we actually train is the classifier head**, which every method must train. The adapters alone are under 0.2% of the model.

---

## Slide 9 — LoRA example: trainable parameters ≠ GPU memory

**On the slide:** which memory components each method needs, and a byte count for the $1024\times1024$ matrix (16 MiB vs ≈ 4.13 MiB, ≈ 3.9×).

**The rule of thumb behind the 3.9×.** With Adam in FP32, full fine-tuning stores **four numbers per weight**: the weight, its gradient, and Adam's $m$ and $v$ (16 bytes). LoRA removes the last three for frozen weights but must keep the weight (4 bytes). So:

- on weight-related memory, the saving can never exceed **4×**, however small $r$ is;
- the frozen $W_0$ sets a floor, and $r$ only decides how close to 4× you get ($16/4.125\approx3.88$ at $r=4$).

**Why real runs save even less.** Activations are needed by both methods for backpropagation, and they scale with batch size × sequence length × hidden size × layers. They add the same amount to both sides, pulling the ratio toward 1×. Mixed precision (FP16/BF16 weights, often with FP32 master copies) changes the byte counts but not this reasoning.

**The same pattern in the LoRA paper.** On GPT-3, LoRA reduces trainable parameters about **10,000×** but GPU memory only about **3×**. Our experiment (slide 13, ≈ 2× less peak memory) fits the same pattern.

---

## Slide 10 — O-LoRA: orthogonality and core hypothesis

**On the slide:** the sequential-task setting, the definition of orthogonal subspaces, the test $A_i^\top A_t=0$, and the hypothesis that orthogonal update subspaces interfere less.

**What forgetting looks like.** If task 2 is learned by moving the same weight directions that task 1 relied on, task 1's behavior is overwritten. Nothing in plain LoRA prevents this.

**Why checking columns is enough.** Every vector in $\operatorname{col}(A_i)$ is $A_ia$, and every vector in $\operatorname{col}(A_t)$ is $A_tb$, so
$$(A_ia)^\top(A_tb)=a^\top(A_i^\top A_t)\,b=0\ \ \forall a,b\iff A_i^\top A_t=0 .$$
Checking $r^2$ column pairs replaces checking infinitely many vectors.

**Why $A$ and not $B$.** $sA_tB_tx=A_t(sB_tx)$ is always a combination of $A_t$'s columns, whatever $x$ is. So $\operatorname{col}(A_t)$ is the set of **output directions** task $t$ can change. $B_t$ only decides *how much* of each direction to use for a given input.

**Why the squared Frobenius norm.** $\|A_i^\top A_t\|_F^2=\sum_{p,q}(A_i^\top A_t)_{pq}^2$ sums every pairwise overlap into one smooth, differentiable number that is zero only for exact orthogonality, which makes it suitable as a loss term.

**Limits of the idea.**
- **Capacity.** At most $\lfloor d/r\rfloor$ mutually orthogonal rank-$r$ subspaces fit in $\mathbb R^d$: 96 for $d=768$, $r=8$.
- **Heuristic, not proof.** Orthogonal output directions in one layer do not guarantee zero interference; later layers mix directions, and inputs from different tasks overlap.

---

## Slide 11 — O-LoRA: method

**On the slide:** the accumulated weight $W_t$, the loss $\mathcal L_t=\mathcal L_{\rm task}+\lambda\sum_i\|A_i^\top A_t\|_F^2$, and properties (no replay, no task ID, no guarantee, growing storage).

**Reading the loss.**
- $\mathcal L_{\rm task}$ is computed with *all* earlier updates in place, so the new task is learned on top of the old ones.
- $\lambda$ is a trade-off: larger values force more orthogonality but leave less freedom to fit the new task; too small and the penalty does nothing.
- Only $A_t$ is penalized; $B_t$ is unconstrained.

**Why "no task ID" matters: contrast with separate adapters.**

| | One LoRA per task, kept separate | O-LoRA |
|---|---|---|
| Interference | none (never combined) | reduced by the penalty |
| Inference | must know the task to pick the adapter | one merged model, no task label |
| Old data | not needed | not needed |

**Why "no replay".** Old tasks are protected through their frozen $A_i$ matrices, not stored examples. This helps where privacy or storage rules out keeping old data.

**Footer detail.** As in our experiment, the paper applies LoRA to the attention query and value matrices.

### Background for Q&A: LoRA vs. O-LoRA (no longer a slide)

| Aspect | LoRA | O-LoRA |
|---|---|---|
| Goal | adapt with few trainable parameters | learn tasks in sequence with limited interference |
| Trained | $A,B$ for one task | current $A_t,B_t$; earlier factors frozen |
| Loss | task loss | task loss + $\lambda\cdot$ orthogonality penalty |
| Forgetting | no mechanism | reduced, not guaranteed |
| Storage | one adapter per kept task | grows with the number of tasks |

The two papers use different models, data and settings, so their numbers cannot be compared directly; the table compares design only. We did not run O-LoRA.

---

## Slide 12 — Experiment setup

**On the slide:** model and data; split, preprocessing and training settings; checkpoint selection; a LoRA vs. full fine-tuning table (what is trained, 0.8944%, learning rates, LoRA config).

**Why each choice.**

| Choice | Reason |
|---|---|
| **Stratified** 10% validation split | keeps the four classes balanced in validation, like the test set |
| Test split used **once** | no decision is tuned on it, so the reported numbers are unbiased |
| Select by **macro F1** | weights every class equally; on balanced data it is almost the same as accuracy |
| Batch 8 × 4 accumulation | an effective batch of 32 that fits in limited GPU memory |
| BF16/FP16 | halves activation and weight memory; BF16 when supported, as it is more numerically robust |
| Linear schedule, 6% warmup | avoids large early updates while the new classifier head is still random |

**Why the learning rates differ (10×).** LoRA's $B$ starts at zero and there are few trainable numbers, so they need a larger step (≈ $10^{-4}$). Full fine-tuning uses a small step (≈ $10^{-5}$) so as not to wreck pretrained features. Both are standard, but the comparison now changes **two** things at once (slide 14).

**What 0.8944% contains.** See the breakdown under slide 8: 147,456 adapter + 593,668 classifier parameters, out of ≈ 83 M. Most of it is the head.

**Model.** DistilRoBERTa is RoBERTa-base distilled to 6 layers (hidden size 768). AG News (Zhang et al.) has 1,900 test articles per class.

**Footer note.** The notebook sets $r=8$, but the saved LoRA result files do not record the rank, so the rank comes from the configuration, not the results.

---

## Slide 13 — Results on the AG News test split

**On the slide:** a comparison table (accuracy, macro F1, loss, wall time, peak memory), the per-class table for LoRA, the error count and the main confusion.

**Putting the 1.26 pp gap in perspective.**
- Full fine-tuning: $7{,}600\times(1-0.950263)\approx$ **378 errors**. LoRA: **474 errors**.
- So LoRA makes about **96 more mistakes, ≈ 25% more**, even though the accuracy gap looks small.
- Exact gaps: accuracy 1.2631 pp, macro F1 1.2661 pp. Subtracting the *rounded* table values gives 1.27 for both.

**Why the resource savings are modest (links to slides 6 and 9).**
- **Time:** 47.31 vs. 66.38 min is ≈ 29% faster, not 100×, because the forward pass and the activation gradients through the frozen model still run.
- **Memory:** 1.099 vs. 2.138 GiB peak allocated is ≈ 49% less, because the frozen weights and the activations remain.
- This is the LoRA paper's "10,000× parameters, 3× memory" pattern again.

**Allocated vs. reserved.** *Allocated* is memory held by live tensors. *Reserved* is what PyTorch's caching allocator has claimed from the GPU, including cached free blocks. Reserved is what other processes cannot use.

**Reading the per-class table.**
- 139 Business→Sci/Tech plus 108 Sci/Tech→Business = 247 errors, **52% of all 474**.
- That one confusion explains the two weakest numbers: Business has the lowest **recall** (0.897) because its articles leak into Sci/Tech, and Sci/Tech has the lowest **precision** (0.902) because it receives them.
- Many such articles are genuinely ambiguous (a tech company's earnings, a telecom deal), so part of the error may be label ambiguity rather than model weakness.
- Sports is easiest because its vocabulary is distinctive.
- Macro F1 ≈ accuracy because the test set is balanced (1,900 per class).

---

## Slide 14 — Limitations

**On the slide:** six limits of our evidence and four situations where LoRA may not be best.

**What would fix each limit.**

| Limit (on slide) | Why it matters | What would fix it |
|---|---|---|
| One run per method | a 1.26 pp gap may be partly seed noise | 3–5 seeds per method, report mean ± spread |
| Different learning rates | the gap mixes method and learning-rate effects | tune the learning rate for each method separately |
| No rank comparison | cannot say whether a larger $r$ closes the gap | repeat with $r=4,8,16$ |
| ≈ 30 s evaluation runtimes | dominated by data loading and batching; not latency | time single predictions, merged vs. unmerged |
| $1024\times1024$ example | one matrix, not a whole-model memory figure | measure memory per component in the real run |
| O-LoRA not evaluated | slides 10–11 rest on the paper only | run a task sequence with and without the penalty |

**The common thread in "when LoRA may not be best".** All four come back to the low-rank hypothesis or to practicality:
- **Broad behavior change** (a new language, a domain far from pretraining): the needed $\Delta W$ may not be low rank.
- **Rank misses the quality target:** raising $r$ erodes the savings; at some point full fine-tuning is simpler.
- **Small model, ample compute:** e.g. our full fine-tuning needed only ≈ 2.1 GiB, so LoRA's savings matter little, and full fine-tuning has fewer hyperparameters (no rank, $\alpha$ or target modules).
- **Deployment:** separate adapters suit multi-task serving; merged weights suit one fast model.

---

## Wrap-up — answering the guiding question (spoken, no slide)

At the end of slide 14, the speaker answers the question from slide 2:

| Part of the question | Answer | Evidence |
|---|---|---|
| How much fewer trainable parameters? | 128× for one $1024\times1024$ matrix at $r=4$; 0.8944% of DistilRoBERTa trained in our run (mostly the head) | slides 8, 12 |
| Quality? | 1.26 pp lower accuracy (≈ 25% more errors) | slide 13 |
| Resources? | ≈ 29% faster training, ≈ half the peak memory: real, but far smaller than the parameter ratio | slides 9, 13 |
| Sequential tasks? | O-LoRA adds per-task LoRAs with an orthogonality penalty | slides 10–11 |
| What is still missing? | repeated runs, matched learning rates, rank sweep, latency | slide 14 |

**Take-away:** LoRA fits when limiting trained parameters matters (many task variants, limited GPU memory, very large models) **and** the measured quality trade-off is acceptable.

---

## Sources — where each citation appears

| Work | Footer of | Used for |
|---|---|---|
| Li et al. (ICLR 2018) | slides 4, 5 | intrinsic dimension via random-subspace training |
| Aghajanyan et al. (ACL 2021) | slides 4, 5 | low intrinsic dimension of LM fine-tuning (≈ 200 parameters for 90% on MRPC) |
| Oymak et al. (arXiv 2019) | slide 5 | low-rank Jacobian structure |
| Hu et al. (ICLR 2022) | slides 5–7 | LoRA; GPT-3 figures (10,000× parameters, 3× memory, 350 GB → 35 MB checkpoints) |
| Wang et al. (Findings of EMNLP 2023) | slides 10, 11 | O-LoRA |
| Zhang et al. (NeurIPS 2015) | slide 12 | AG News benchmark |

---

## Appendix — Likely questions and short answers

**Q: Why apply LoRA only to query and value?** Hu et al. found that adapting $W_q$ and $W_v$ gives most of the benefit for a fixed parameter budget, and O-LoRA uses the same targets. Adapting more modules (keys, output projection, MLP) is possible and increases the parameter count.

**Q: Why does memory not drop by 100× if parameters do?** With Adam, at most ≈ 4× can be saved on weight-related memory, because the frozen weights stay. Activations, the same for both methods, narrow it further (slide 9).

**Q: Does LoRA slow down inference?** Not if merged: $W_0+sAB$ is a single matrix of the original size. Unmerged adapters add a small extra product per adapted layer. We did not measure it.

**Q: How do you choose $r$?** Empirically, usually 4–64: start small and increase until validation quality stops improving. We used $r=8$ and did not sweep.

**Q: Is the 1.26 pp gap significant?** We cannot say from one run per method with different learning rates (slide 14). In error counts it is 474 vs. ≈ 378.

**Q: Why is most of the 0.8944% the classifier?** The new head ($\approx$ 594 K parameters) must be trained by every method and is larger than all twelve rank-8 adapters together (≈ 147 K).

**Q: How is O-LoRA different from training several independent LoRAs?** Independent LoRAs need a task label at inference to choose the adapter. O-LoRA merges everything into one model and uses the penalty so the merged updates interfere less.

**Q: What happens when there are more tasks than fit orthogonally?** At most $\lfloor d/r\rfloor$ orthogonal rank-$r$ subspaces exist in $\mathbb R^d$ (96 for $d=768$, $r=8$). Beyond that, overlap is unavoidable and the penalty cannot reach zero.
