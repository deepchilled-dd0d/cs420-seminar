# Slide-by-Slide Explanation — LoRA and O-LoRA

A detailed companion to `presentation.tex` (18 slides). For each slide it explains what is shown, why it is true, and how it connects to the rest of the talk, with likely audience questions where useful. Slide numbers follow the compiled deck. Speakers: Quan (slides 1–6), Tinh (7–10), Lac (11–18).

**Notation used throughout**

| Symbol | Meaning |
|---|---|
| $W_0\in\mathbb R^{d\times k}$ | a pretrained weight matrix (frozen under LoRA) |
| $x\in\mathbb R^{k}$, $h\in\mathbb R^{d}$ | layer input and layer output; $h=W_0x$ when frozen |
| $\Delta W$ | the change to $W_0$ learned during fine-tuning |
| $A\in\mathbb R^{d\times r}$, $B\in\mathbb R^{r\times k}$ | LoRA factors, $\Delta W=\tfrac{\alpha}{r}AB$ |
| $r$ | LoRA rank, $r\ll\min(d,k)$ |
| $\alpha$, $s=\alpha/r$ | scaling hyperparameter and resulting scale |
| $\Phi_0$, $\Theta$ | all pretrained parameters; the trainable LoRA parameters |
| $D$, $d$, $d_{\rm int}$ | full parameter dimension, subspace dimension, intrinsic dimension (slide 4 only) |
| $A_t,B_t$, $\lambda$ | task-$t$ LoRA factors in O-LoRA; penalty weight |

> **Naming convention.** The deck and report put $A$ on the output side ($d\times r$) and $B$ on the input side ($r\times k$), with $A$ Gaussian and $B=0$ at initialization. The original LoRA paper and the PEFT library use the opposite letters ($\Delta W = BA$, $B=0$). The mathematics is identical; only the names are swapped. The footnote on slide 6 says this.

> **Two different "$d$"s.** On slide 4, $d$ is the dimension of a random search subspace. From slide 6 onward, $d$ is the number of rows of $W_0$. They are unrelated.

---

## Slide 1 — Title

Title, authors (Thien-Lac Quach, Minh-Quan Pham Nguyen, Tinh Nguyen Van), CS420 Seminar Group 4, 23 November 2026.

---

## Slide 2 — Guiding question and outline

**Guiding question:** *How much does LoRA reduce trainable parameters, and what evidence is needed to judge its quality and resource use?*

The question deliberately has two parts with different kinds of answers:

1. **"How much does it reduce trainable parameters?"** This is answered analytically. For one $d\times k$ matrix the count goes from $dk$ to $r(d+k)$ (slides 6 and 9).
2. **"What evidence is needed to judge quality and resource use?"** This cannot be derived. It needs measurement: accuracy, training time and GPU memory, and an honest statement of what one experiment can and cannot show (slides 10, 14–16).

The outline maps onto the talk: background (3–5) → method (6–8) → worked example (9–10) → O-LoRA (11–13) → experiment (14–15) → trade-offs and conclusion (16–17).

---

## Slide 3 — Ways to adapt a pretrained model

The figure shows the usual pipeline: pre-train a large model on generic data, then adapt it to a downstream task. The table lists five ways to adapt, ordered roughly by *what* is trained.

| Method | Formula | What is trained | Notes |
|---|---|---|---|
| Linear probing | $\hat y = W_{\rm head}\,f_{W_0}(x) + b$ | only a new head $W_{\rm head}, b$ | The backbone $f_{W_0}$ is a fixed feature extractor. Cheapest, but the features cannot adapt, so it often underperforms when the new task differs from pre-training. |
| Full fine-tuning | $h=(W_0+\Delta W)x$ | every weight | Highest capacity. Needs gradients and optimizer state for every parameter, and a full-size checkpoint per task. |
| LoRA | $h = W_0x + \frac{\alpha}{r}ABx$ | low-rank factors $A,B$ | Changes how each adapted layer computes, at a tiny parameter cost; can be merged after training (slide 8). |
| Adapter | $h' = h + W_{\rm up}\,\sigma(W_{\rm down}h)$ | small bottleneck MLP inserted after a layer | $W_{\rm down}$ projects to a small dimension, $\sigma$ is a nonlinearity, $W_{\rm up}$ projects back; the residual "$h+$" keeps the original path. Because of $\sigma$ it **cannot** be merged into $W_0$, so it adds inference latency. |
| Prompt / prefix tuning | $X'=[P;X]$ / $h=\operatorname{Attn}(Q,[P_K;K],[P_V;V])$ | trainable vectors $P$ | Prompt tuning prepends learned "soft tokens" to the input embeddings $X$. Prefix tuning prepends learned key/value vectors in every attention layer. Weights are untouched; the prefix uses up part of the sequence length. |

**Key point.** LoRA is in the middle. Unlike linear probing it changes the internal computation. Unlike full fine-tuning it trains very few parameters. Unlike adapters, its update is linear, so it can be folded into $W_0$ with no inference cost.

---

## Slide 4 — Intrinsic dimension of an objective

This slide explains the idea behind LoRA, from Li et al. (2018) and Aghajanyan et al. (2021).

### Panel (a): what intrinsic dimension means

- The full parameter space has $D$ dimensions (here $D=3$, axes $\theta_1,\theta_2,\theta_3$).
- Training moves from the starting point $\theta_0^{(D)}$ to a solution $\theta_*^{(D)}$.
- In practice the solutions do not form a single point. They form a **solution set** $\mathcal S$ of dimension $s$: many parameter settings are equally good. In the picture $\mathcal S$ is a line, so $s=1$.
- The **intrinsic dimension** is the codimension of $\mathcal S$:
  $$D = d_{\rm int} + s \quad\Longrightarrow\quad d_{\rm int} = 3-1 = 2.$$
  Intuitively, moving along $\mathcal S$ changes nothing, so only $d_{\rm int}$ directions actually matter for reaching it.

### Panel (b): how it is measured

- Fix the start $\theta_0^{(D)}$ and a **random** projection $P\in\mathbb R^{D\times d}$.
- Optimize only a small vector $\theta^{(d)}\in\mathbb R^{d}$; the full parameters are always
  $$\theta^{(D)} = \theta_0^{(D)} + P\,\theta^{(d)}.$$
  So the search is restricted to a $d$-dimensional affine subspace (the grey plane).
- A $d$-dimensional subspace and an $s$-dimensional solution set in $D$-space generally intersect when $d + s \ge D$, that is, when $d\ge d_{\rm int}$. Here the plane ($d=2$) crosses the line $\mathcal S$ at $\theta_*^{(D)}$, so the problem is solvable inside the plane.
- In practice, $d_{\rm int}$ is estimated as the smallest $d$ that reaches a set fraction (e.g. 90%) of full-training performance.

### Why it matters

Aghajanyan et al. found that fine-tuning pretrained language models has a **very low** intrinsic dimension. A random subspace of a few hundred to a few thousand dimensions can recover most of full fine-tuning performance, even though the model has hundreds of millions of parameters. Larger pretrained models tend to have *lower* intrinsic dimension. This suggests that the fine-tuning update lives in a small subspace, which is the starting point for LoRA.

---

## Slide 5 — Why a low-rank update?

This slide states the LoRA hypothesis and how it differs from the intrinsic-dimension result.

1. **Low-rank structure is common.** Oymak et al. (2019) showed that the Jacobians of trained networks are approximately low rank. Hu et al. (2022) observed that learned updates have a few dominant directions.
2. **Fine-tuning has low intrinsic dimension** (slide 4): $d\ll D$ suffices.
3. **The two are not the same.**
   - *Intrinsic dimension* counts free coordinates of the **whole** parameter vector, searched through one random global projection $P$.
   - *Rank* is a property of **one matrix** $\Delta W$: the dimension of its column space.
   - A low $d_{\rm int}$ does not mathematically imply that each $\Delta W$ is low rank, and vice versa. Low intrinsic dimension *motivates* the low-rank assumption but does not *prove* it.
4. **LoRA hypothesis:** the task-specific change to each weight matrix has low rank,
   $$\operatorname{rank}(\Delta W)\ll\min(d,k).$$
5. **Consequence.** If the hypothesis holds, we lose little by forcing $\Delta W$ to be low rank. Any matrix of rank $\le r$ can be written as a product $AB$ with $A\in\mathbb R^{d\times r}$, $B\in\mathbb R^{r\times k}$, so we learn $A$ and $B$ instead of $\Delta W$.

**Possible question:** *"Is the hypothesis true?"* — It is an empirical claim. Hu et al. report that very small ranks (even $r=1$–$8$ on GPT-3's attention matrices) match full fine-tuning on many benchmarks. Our experiment does not test it directly, because we ran only one rank (slide 16).

---

## Slide 6 — LoRA formulation

$$h = W_0x + \frac{\alpha}{r}ABx,\qquad W_0\in\mathbb R^{d\times k}\ (\text{frozen}),\ A\in\mathbb R^{d\times r},\ B\in\mathbb R^{r\times k},\ x\in\mathbb R^{k},\ r\ll\min(d,k).$$

**Shapes.** $Bx\in\mathbb R^{r}$ (compress), $A(Bx)\in\mathbb R^{d}$ (expand), and $W_0x\in\mathbb R^{d}$, so the two terms can be added. The product $AB$ is $d\times k$, the same shape as $W_0$, but $\operatorname{rank}(AB)\le r$.

**Objective.** Full fine-tuning solves
$$\min_{\Phi}\ -\sum_{(x,y)\in\mathcal D}\log P_{\Phi}(y\mid x).$$
LoRA keeps the pretrained $\Phi_0$ fixed and optimizes only $\Theta$, which defines the update $\Delta\Phi(\Theta)$:
$$\min_{\Theta}\ -\sum_{(x,y)\in\mathcal D}\log P_{\Phi_0+\Delta\Phi(\Theta)}(y\mid x),\qquad |\Theta|\ll|\Phi_0|.$$
$\Theta$ is the collection of all $(A,B)$ pairs, one per adapted matrix.

**Rank and scaling.**
- $r$ sets capacity: the update can move $W_0$ in at most $r$ independent output directions.
- The factor $\alpha/r$ keeps the size of the update roughly stable when $r$ changes. Without it, a larger $r$ (more summed terms) would give a larger update at the same learning rate, so the learning rate would need retuning each time. With it, one can usually fix $\alpha$ (often $\alpha=2r$ or a constant such as 16) and sweep $r$.

**Initialization.** $A\sim$ Gaussian, $B=0$, so $AB=0$ and at step 0 the model computes exactly $W_0x$. Training therefore starts from the pretrained model rather than a randomly perturbed one. Only one factor can be zero: if both were zero, both gradients would be zero ($\partial/\partial A\propto (\cdot)(Bx)^\top=0$ and $\partial/\partial B\propto A^\top(\cdot)=0$) and nothing would ever train. With $A\ne0$, $B$ gets a nonzero gradient at step 1, and $A$ starts receiving gradient once $B$ becomes nonzero.

**Cost per adapted matrix.**
$$\text{full: } dk \qquad\text{LoRA: } |A|+|B| = dr + rk = r(d+k).$$
Gradients and optimizer states (e.g. Adam's $m,v$) exist only for trainable parameters, so they shrink by the same ratio. The stored per-task checkpoint shrinks too: only $A,B$ (and any trained head) need saving.

---

## Slide 7 — LoRA training: only $A$ and $B$ receive gradients

The diagram follows one adapted layer during training:

1. **Two paths from the same input $x$.** The frozen path computes $W_0x$. The adapter path computes $Bx\in\mathbb R^r$ (bottleneck), then $A(Bx)\in\mathbb R^d$.
2. **Scale and add.** $h = W_0x + s\,ABx$ with $s=\alpha/r$.
3. **Backward pass.** The loss gradient flows back through $h$, so we get $\partial\mathcal L/\partial h$ as usual. From it:
   $$\frac{\partial\mathcal L}{\partial A} = s\,\frac{\partial\mathcal L}{\partial h}(Bx)^\top,\qquad \frac{\partial\mathcal L}{\partial B} = s\,A^\top\frac{\partial\mathcal L}{\partial h}\,x^\top.$$
   The gradient with respect to $x$ still flows through both paths ($W_0^\top$ and $sB^\top A^\top$) to earlier layers. However, no gradient **for $W_0$** is computed or stored, and the optimizer never updates it.
4. **$W_0$ is still used.** It must be in memory and is multiplied in every forward pass (and its transpose in the backward pass). LoRA removes $W_0$'s *training* cost (gradient, optimizer state) but not its *storage* or *compute* cost. Slide 10 builds on this.

In the experiment, LoRA dropout (0.05) is applied to $x$ on the adapter path only, as regularization.

---

## Slide 8 — After training: separate adapter or merged weights

Because the LoRA update is linear in $x$:
$$W_0x + sABx = (W_0 + sAB)\,x.$$

**Option 1 — keep the adapter separate (left).**
- Inference computes both paths, $W_0x + sABx$: a small extra matrix product per adapted layer.
- One base model in memory can serve many tasks, each with a few-MB adapter, and switching tasks means swapping adapters.
- An adapter is only valid for the exact base model it was trained on; it is meaningless without the matching $W_0$.

**Option 2 — merge (right).**
- Compute once: $W_{\rm merged} = W_0 + sAB$.
- The layer is again a single $d\times k$ matrix multiplication, identical in cost to the original, so there is **no extra inference latency** from LoRA. Adapters (slide 3) cannot do this because of their nonlinearity $\sigma$.
- To switch tasks you can unmerge by subtracting $sAB$ (up to floating-point rounding) or keep a copy of $W_0$.

**Trade-off.** Separate adapters suit multi-task serving; merging suits serving one specialized model as fast as possible.

---

## Slide 9 — Worked example: counting trainable parameters

**Setup:** one matrix $W_0\in\mathbb R^{1024\times1024}$, so $d=k=1024$. Weights only: no biases, no classifier.

**Full fine-tuning:** $N_{\rm full} = dk = 1024^2 = 1{,}048{,}576$.

**LoRA:** $|A| = 1024r$, $|B| = 1024r$, so $N_{\rm LoRA} = r(d+k) = 2048r$.

**Example $r=4$:** $|A|=|B|=4{,}096$, $N_{\rm LoRA}=8{,}192$.
$$\text{ratio} = \frac{1{,}048{,}576}{8{,}192} = 128,\qquad \text{reduction} = 1-\frac{1}{128} = 0.9921875 = 99.21875\%.$$

**General formula for a square $n\times n$ matrix:**
$$\text{ratio} = \frac{dk}{r(d+k)} = \frac{n^2}{2nr} = \frac{n}{2r} = \frac{512}{r},\qquad \text{reduction} = 1-\frac{r}{512}.$$

| $r$ | $\lvert A\rvert=\lvert B\rvert$ | $N$ | Ratio | Reduction |
|---|---|---|---|---|
| full | — | 1,048,576 | $1\times$ | 0% |
| 4 | 4,096 | 8,192 | $128\times$ | 99.21875% |
| 8 | 8,192 | 16,384 | $64\times$ | 98.4375% |
| 16 | 16,384 | 32,768 | $32\times$ | 96.875% |

**Observations.**
- $N_{\rm LoRA}$ is linear in $r$: doubling $r$ doubles the cost and halves the ratio.
- The ratio $n/(2r)$ grows with $n$. LoRA saves *more* on larger matrices, which is why it is attractive for very large models (for GPT-3, $n=12{,}288$, so $r=4$ gives a ratio of 1,536 per matrix).
- LoRA only saves parameters when $r(d+k)<dk$, i.e. $r < dk/(d+k)$ = 512 here. In practice $r$ is far below this.
- This counts **one matrix**. A whole model adapts many matrices and usually also trains a task head, so the whole-model percentage differs. In our experiment it is 0.8944% (slide 14).

---

## Slide 10 — Worked example: trainable parameters ≠ GPU memory

**What training must hold in memory:**

| Component | Full FT | LoRA |
|---|---|---|
| Original $W_0$ | trainable | frozen, but **kept** |
| Adapter $A,B$ | — | trainable |
| Gradients | for $W_0$ | for $A,B$ only |
| Adam $m,v$ | for $W_0$ | for $A,B$ only |
| Activations | required | required |
| Temporary tensors | required | required |

LoRA saves only the rows that depend on *trainable* parameters: gradients and optimizer state. Weights, activations and workspace remain.

**Byte count** for the $1024\times1024$ matrix, $r=4$, FP32 (4 bytes/value), Adam (2 states per trainable value), activations excluded:

| Memory | Full FT | LoRA $r=4$ |
|---|---|---|
| Weights | $1{,}048{,}576\times4$ B $=4$ MiB | $4$ MiB $+$ $8{,}192\times4$ B $=4$ MiB $+$ 32 KiB |
| Gradients | 4 MiB | 32 KiB |
| Adam $m$ and $v$ | $2\times4$ MiB $=8$ MiB | $2\times32$ KiB $=64$ KiB |
| **Total** | **16 MiB** | **4 MiB + 128 KiB = 4.125 MiB ≈ 4.13 MiB** |

Reduction: $16/4.125\approx3.88\approx3.9\times$, compared with $128\times$ for trainable parameters.

**Why the gap?** Full fine-tuning costs about $4\times$ the weight size (weights + gradient + 2 Adam states). LoRA removes the last three for $W_0$, so the best possible saving on this matrix approaches $4\times$, no matter how small $r$ is. The frozen $W_0$ sets a floor.

**Activations make it narrower still.** Activations scale with batch size × sequence length × hidden size and are needed by both methods for backpropagation. In a real run they can be a large share of memory, which pushes the measured ratio further toward $1\times$. This is why the experiment shows memory roughly **halved** (slide 15) rather than cut by $100\times$. Mixed precision (BF16/FP16), which the experiment uses, changes the byte counts but not this reasoning.

---

## Slide 11 — O-LoRA: orthogonality and the core hypothesis

### Setting: continual learning

Tasks $\mathcal D_1,\dots,\mathcal D_T$ arrive one at a time, and at step $t$ only $\mathcal D_t$ is available (no stored data from earlier tasks). Training on task $t$ can overwrite what was learned for tasks $1,\dots,t-1$. This is **catastrophic forgetting**.

### Orthogonal subspaces

- Two subspaces $\mathcal U,\mathcal V\subseteq\mathbb R^d$ are orthogonal if $u^\top v=0$ for every $u\in\mathcal U$, $v\in\mathcal V$.
- Let $\mathcal U_i=\operatorname{col}(A_i)$, the span of the $r$ columns of $A_i\in\mathbb R^{d\times r}$. Every vector in it is $A_i a$ for some $a\in\mathbb R^r$; likewise $\mathcal U_t$. Then
  $$(A_ia)^\top(A_tb) = a^\top(A_i^\top A_t)\,b\quad\forall a,b \;=\;0 \iff A_i^\top A_t = 0\in\mathbb R^{r\times r}.$$
  So checking the $r^2$ pairs of columns is enough; no need to check every vector.
- Entry $(A_i^\top A_t)_{pq}$ is the inner product of column $p$ of $A_i$ with column $q$ of $A_t$. Hence
  $$\|A_i^\top A_t\|_F^2 = \sum_{p,q}\big((A_i^\top A_t)_{pq}\big)^2 \ \ge 0,$$
  which is zero exactly when every pair of columns is orthogonal. It is a smooth, differentiable measure of overlap, suitable as a loss term.

### O-LoRA hypothesis

- The update for task $t$ is $\tfrac{\alpha}{r}A_tB_tx = A_t\big(\tfrac{\alpha}{r}B_tx\big)$, a linear combination of $A_t$'s columns. So whatever $x$ is, the update **always lies in $\operatorname{col}(A_t)$**. That column space is taken as an approximation of task $t$'s *update subspace*: the low-dimensional set of output directions in which task $t$ changes the layer (motivated by the low-rank structure of gradient updates).
- If task $t$'s update directions are orthogonal to those of earlier tasks, then for the outputs that earlier tasks care about, the new update adds a component in directions they did not use, so it should **interfere less**.
- Method in one line: one LoRA per task, subspaces kept approximately orthogonal.

**Caveat.** Orthogonality of output directions is a heuristic for less interference, not a proof of zero interference. Later layers mix directions, and inputs from different tasks overlap. This is why slide 12 says "encourages, not guarantees".

---

## Slide 12 — O-LoRA: method

**Accumulated weight.** Each task $i$ has its own $(A_i,B_i)$. After learning tasks $1..t$:
$$W_t = W_0 + \sum_{i=1}^{t}\tfrac{\alpha}{r}A_iB_i.$$
While training task $t$, $W_0$ and $\{A_i,B_i\}_{i<t}$ are frozen; only $A_t,B_t$ train.

**Loss for task $t$:**
$$\mathcal L_t = \underbrace{-\sum_{(x,y)\in\mathcal D_t}\log P_{\Phi_t}(y\mid x)}_{\mathcal L_{\rm task}} \;+\; \lambda\sum_{i=1}^{t-1}\big\|A_i^\top A_t\big\|_F^2,\qquad \lambda>0.$$
- The first term fits the current task with all earlier updates in place.
- The second term penalizes overlap between $\operatorname{col}(A_t)$ and every earlier $\operatorname{col}(A_i)$.
- $\lambda$ trades off the two: larger $\lambda$ gives more orthogonality but less freedom to fit task $t$.
- Only $A_t$ appears in the penalty, because $A$ defines the output subspace; $B_t$ is unconstrained.

**Properties.**
- **No replay data.** Earlier tasks are "remembered" through their frozen factors $A_i$, not through stored examples. This helps where privacy or storage rules out keeping old data.
- **No task ID at inference.** All updates are summed into $W_t$, so one model handles inputs from any task. (Methods that keep a separate adapter per task must be told which adapter to use.)
- **Constraint is soft.** With finite $\lambda$, $A_i^\top A_t$ is small but generally not zero, so neither perfect orthogonality nor zero forgetting is guaranteed.
- **Storage grows with $T$.** The $(A_i,B_i)$ factors are kept to compute future penalties. Also, at most $d/r$ mutually orthogonal rank-$r$ subspaces fit in $\mathbb R^d$, so the approach has a capacity limit for very long task sequences.
- In Wang et al. (2023), LoRA is applied to the attention **query and value** matrices, the same targets as in our experiment.

---

## Slide 13 — LoRA and O-LoRA compared

| Aspect | LoRA | O-LoRA |
|---|---|---|
| Objective | adapt with few trainable parameters | learn sequential tasks with limited interference |
| Trained | low-rank $A,B$ for one task | current $A_t,B_t$; earlier factors frozen |
| Loss | task loss | task loss $+\ \lambda\cdot$ orthogonality penalty |
| Forgetting | no explicit mechanism | aims to mitigate; no guarantee |
| Storage | one small adapter per task you keep | grows with number of tasks |
| Inference | chosen adapter may be merged | updates accumulated into weights; factors kept for further learning |

**How to read it.** O-LoRA is LoRA plus two things: (1) a per-task structure where earlier adapters are frozen, and (2) a regularizer that pushes new adapters into unused directions. Training plain LoRA on task 2 starting from a model with task 1 merged has no such protection, because nothing stops the new update from overwriting task 1's directions.

**Caution on the last line of the slide.** The LoRA and O-LoRA papers use different base models, datasets and settings, so their reported numbers cannot be compared directly. The table compares *design*, not *performance*.

---

## Slide 14 — Experiment setup

**Model and data.** `distilroberta-base` (a 6-layer distilled RoBERTa, hidden size 768) with a sequence-classification head, on AG News (Zhang et al., 2015): news articles in four balanced classes — World, Sports, Business, Sci/Tech. The test split has 7,600 examples (1,900 per class).

**Protocol (identical for both runs).**
- **Validation:** a stratified 10% of the official training split, seed 42. Stratified means each class keeps its proportion. Used for model selection.
- **Test:** the official 7,600-example split, evaluated **once**, at the end, so it is never used for tuning.
- Max length 256 tokens; mixed precision (BF16 if supported, else FP16) on one GPU.
- 3 epochs; per-device batch 8 with 4 gradient-accumulation steps, so effective batch 32.
- AdamW, weight decay 0.01, linear learning-rate decay with warmup over the first 6% of steps.
- Evaluate each epoch; keep the checkpoint with the best **validation macro F1**.

**What differs.**

| | LoRA | Full fine-tuning |
|---|---|---|
| Trained | query/value LoRA adapters + classifier | entire model |
| Share trained | 0.8944% | 100% |
| Learning rate | $2\times10^{-4}$ | $2\times10^{-5}$ |
| LoRA config | $r=8$, $\alpha=16$ ($s=2$), dropout 0.05 | — |

**Where 0.8944% comes from.** LoRA parameters: 6 layers × 2 matrices (query, value) × $r(d+k)=8(768+768)=12{,}288$ → **147,456**. The classification head is newly initialized and must be trained in both methods: dense $768\times768+768$ plus output $768\times4+4$ → **593,668**. Trainable total 741,124, which PEFT reports as 0.8944% of all parameters. Most of the trainable budget is actually the classifier head, not the adapters.

**Why the learning rates differ.** LoRA factors start at or near zero and are few, so they usually need a larger learning rate (around $10^{-4}$). Full fine-tuning uses a small rate (around $10^{-5}$) to avoid destroying pretrained features. Each is a standard choice for its method, but it means the comparison is not a single controlled variable (slide 16).

**Macro F1.** Compute F1 $=2PR/(P+R)$ for each class and average the four values without weighting. On a balanced test set it is very close to accuracy, which is why the two numbers on slide 15 nearly coincide.

**Provenance note (footer).** The notebook config says $r=8$, but the saved LoRA result files do not record the rank. The report flags this.

---

## Slide 15 — Results on the AG News test split

### LoRA vs. full fine-tuning

| Measure | LoRA | Full FT | Difference |
|---|---|---|---|
| Accuracy | 93.76% | 95.03% | −1.26 pp |
| Macro F1 | 93.76% | 95.03% | −1.27 pp |
| Test loss | 0.1930 | 0.1812 | +0.012 |
| Train wall time | 47.31 min | 66.38 min | −19.07 min (≈ −29%) |
| Peak allocated | 1.099 GiB | 2.138 GiB | −1.039 GiB (≈ −49%) |
| Peak reserved | 1.369 GiB | 2.297 GiB | −0.928 GiB |

**Interpretation.**
- **Quality.** Training under 1% of the parameters costs about 1.3 points of accuracy here. Whether that is acceptable depends on the application.
- **Time.** LoRA is faster, but not 100× faster. The forward pass and most of the backward pass (gradients with respect to activations, through the frozen $W_0$) are still computed. LoRA mainly skips weight gradients for $W_0$ and the optimizer update for the frozen parameters.
- **Memory.** Peak allocated memory roughly halves, matching slide 10: frozen weights, activations and workspace remain.
- **Allocated vs. reserved.** *Allocated* is memory held by live tensors. *Reserved* is what PyTorch's caching allocator has claimed from the GPU (allocated plus cached free blocks). Reserved is what other processes cannot use.

### LoRA per-class results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| World | 0.955 | 0.942 | 0.949 |
| Sports | 0.982 | 0.988 | 0.985 |
| Business | 0.912 | 0.897 | 0.905 |
| Sci/Tech | 0.902 | 0.923 | 0.912 |

- **Precision** for class $c$: of the items predicted $c$, the fraction that are truly $c$. **Recall**: of the items truly $c$, the fraction found.
- Sports is easiest (distinctive vocabulary). Business and Sci/Tech are hardest.
- **Errors:** 7,600 − 7,126 correct = **474** errors (6.24%). The largest confusion is Business ↔ Sci/Tech: **139** Business articles predicted Sci/Tech and **108** Sci/Tech predicted Business, so 247 of the 474 errors (52%). Many articles (tech companies' earnings, telecom deals) plausibly belong to both, so part of this error may be label ambiguity rather than model weakness.
- This matches the per-class numbers: Business recall (0.897) is the lowest because Business items leak into Sci/Tech, and Sci/Tech precision (0.902) is the lowest because it receives them.

---

## Slide 16 — Limitations and when LoRA may not fit

### Limits of our evidence

| Limitation | Why it matters |
|---|---|
| One run per method | No variance estimate. A 1.26 pp gap could partly be seed noise; repeated runs would give confidence intervals. |
| Different learning rates | The gap mixes the effect of the method with the effect of the learning rate, so it cannot be attributed to LoRA alone. |
| No rank sweep | We cannot say how quality depends on $r$, or whether a larger $r$ would close the gap. |
| Evaluation runtimes (~30 s each) | Not a controlled latency benchmark (data loading and batching dominate); merged vs. unmerged inference was not timed. |
| $1024\times1024$ example | Illustrates one matrix; it is not a whole-model memory measurement. |
| O-LoRA not evaluated | Everything on slides 11–13 comes from the paper, not from our experiments. |

### When LoRA may not be the best choice

- **The task needs broad behavior changes** (e.g. a new language or domain far from pre-training). The low-rank hypothesis may fail, and full fine-tuning's capacity may be needed.
- **The chosen rank misses the quality target.** Raising $r$ adds parameters; at some point another method may be better.
- **Small models with ample compute.** Savings are small in absolute terms, and full fine-tuning is simpler (no extra library, no rank/α/target choices). In our case, DistilRoBERTa full fine-tuning used only ~2.1 GiB.
- **Deployment needs.** Separate adapters ease multi-task switching; merged weights simplify serving one model. The right choice depends on how the model will be served.

---

## Slide 17 — Conclusion

This slide answers the guiding question from slide 2, point by point:

| Part of question | Answer | Evidence |
|---|---|---|
| How much fewer trainable parameters? | Rank-4 LoRA: 8,192 vs. 1,048,576 for one $1024\times1024$ matrix (99.22% fewer, $128\times$). In the whole DistilRoBERTa run, 0.8944% of parameters trained. | Slides 9, 14 |
| Quality? | 93.76% accuracy/macro F1 vs. 95.03% for full FT (−1.26 pp). | Slide 15 |
| Resource use? | 47.31 vs. 66.38 min; 1.099 vs. 2.138 GiB peak allocated. Real savings, but far smaller than the parameter ratio. | Slides 10, 15 |
| Sequential tasks? | O-LoRA adds per-task LoRAs with an orthogonality penalty. | Slides 11–13 |
| What evidence is still missing? | Repeated runs, matched learning rates, rank sweep, merged vs. unmerged latency. | Slide 16 |

**Take-away:** LoRA fits when limiting trained parameters matters (storage of many task variants, limited GPU memory, very large models) **and** the measured quality trade-off is acceptable for the task.

---

## Slide 18 — References

- **Hu et al. (2022)**, ICLR — the original LoRA paper: formulation, initialization, $\alpha/r$ scaling, merging.
- **Wang et al. (2023)**, Findings of EMNLP — O-LoRA: per-task LoRA with orthogonality penalty for continual learning.
- **Li et al. (2018)**, ICLR — introduces intrinsic dimension via random subspace training (slide 4).
- **Aghajanyan et al. (2021)**, ACL — shows that fine-tuning pretrained LMs has low intrinsic dimension (slides 4–5).
- **Oymak et al. (2019)** — low-rank Jacobian structure in neural networks (slide 5).
- **Zhang et al. (2015)**, NeurIPS — introduces the AG News classification benchmark (slide 14).
- **Our CS420 seminar report** — full derivations, tables and result files behind every slide.

---

## Appendix — Likely questions and short answers

**Q: Why apply LoRA only to query and value?** Hu et al. found that adapting $W_q$ and $W_v$ gives most of the benefit for a fixed parameter budget, and O-LoRA uses the same targets. Adapting more modules (keys, output projection, MLP) is possible and increases the parameter count.

**Q: Why does memory not drop by 100× if parameters do?** Frozen weights must still be stored and used, and activations scale with batch and sequence length for both methods. LoRA removes only gradients and optimizer states for frozen weights (slide 10).

**Q: Does LoRA slow down inference?** Not if merged: $W_0+sAB$ is a single matrix of the original size. Unmerged adapters add a small extra matrix product per adapted layer. We did not measure this.

**Q: How do you choose $r$?** Empirically, usually $4$–$64$. Start small and increase until validation quality stops improving. We used $r=8$ and did not sweep.

**Q: Is the 1.26 pp gap significant?** We cannot say from one run per method with different learning rates. That is exactly the limitation on slide 16.

**Q: How is O-LoRA different from training several independent LoRAs?** Independent LoRAs need a task ID at inference to choose the adapter. O-LoRA sums all updates into one model and uses the orthogonality penalty so that the summed updates interfere less.

**Q: What happens when there are more tasks than fit orthogonally?** At most $\lfloor d/r\rfloor$ mutually orthogonal rank-$r$ subspaces exist in $\mathbb R^d$ (e.g. 96 for $d=768$, $r=8$). Beyond that, overlap is unavoidable and the penalty cannot reach zero.
