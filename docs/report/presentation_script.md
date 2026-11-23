# Presentation Script — LoRA and O-LoRA: Parameter-Efficient Fine-Tuning of Large Models

CS420 Seminar, Group 4 — 23 November 2026 — **online presentation**

Slide numbers follow the compiled deck (`presentation.tex`, 18 frames). Target length is about 18 minutes, at roughly 125 spoken words per minute (a little slower than in person, because of audio lag and compression).

| Part | Speaker | Slides | Topic | Suggested time |
|------|---------|--------|-------|----------------|
| 1 | Quan | 1–6 | Motivation, background, LoRA formulation | ≈ 6 min |
| 2 | Tinh | 7–10 | LoRA training, merging, worked examples | ≈ 4.5 min |
| 3 | Lac | 11–18 | O-LoRA, comparison, experiment, conclusion | ≈ 7.5 min |

**How to read this script**

- Plain text is spoken.
- *[Italic brackets]* are stage directions, not spoken.
- **▶ NEXT** marks the moment the slide driver advances to the next slide.

> Note: the speaker comments in `presentation.tex` still use the old assignment (Quan 2–6, Lac 7–12, Tinh 13–17). This script follows the current order: Quan, then Tinh, then Lac.

---

## Online setup

### Roles

| Role | Who | Responsibility |
|---|---|---|
| Slide driver | Quan | Shares the screen for the whole talk and advances on every **▶ NEXT**, including during Tinh's and Lac's parts. One shared screen means no hand-off of screen sharing between speakers. |
| Backup driver | Lac | Has the PDF open and ready to share if Quan's connection drops. |
| Chat monitor | Tinh during Parts 1 and 3; Quan during Part 2 | Watches the meeting chat, notes questions for the Q&A, and messages the speaker privately if the audience reports a problem (no audio, frozen screen). |
| Timekeeper | Lac during Parts 1–2; Tinh during Part 3 | Sends a private "1 min left" message to the current speaker if a part is running long. |

### Before the session (≈ 15 min early)

- [ ] Everyone joins early on a stable (preferably wired) connection, with headphones to avoid echo.
- [ ] Each speaker tests their microphone in the meeting app and says a sentence so the others can confirm the volume.
- [ ] Quan shares **only the PDF window** (`build/presentation.pdf`) in full-screen/presentation mode, not the whole desktop. Turn off notifications (Do Not Disturb) and close chat apps and email.
- [ ] Ask a teammate to confirm the shared slide is readable, including the small "Source:" footer and the slide number in the bottom right.
- [ ] All three keep this script and the PDF open on a second screen or device, so each speaker can follow along without seeing the shared screen.
- [ ] Mute when not speaking. Turn cameras on or off per course rules; if cameras are on, the current speaker's camera stays on.

### During the talk

- **Cursor instead of a laser pointer.** The driver moves the mouse cursor to whatever the speaker names. Many viewers will not see the cursor clearly, so speakers always **say** where to look ("the left panel", "the third row of the table").
- **Asking for the next slide.** Speakers do not need to say "next slide"; the driver follows the **▶ NEXT** markers in this script. If out of sync, the speaker says "Next slide, please" or "Could we go back to slide 9?".
- **Handovers.** The outgoing speaker names the next speaker and mutes. The incoming speaker unmutes **before** the handover line ends, then starts with "Thanks, …" — no "Can you hear me?" (the chat monitor will flag audio problems).

### If something goes wrong

| Problem | What to do |
|---|---|
| Speaker's connection drops | The next speaker in order continues from the current slide using this script. The dropped speaker rejoins and stays muted. |
| Driver's connection drops | Lac starts sharing the PDF from the current slide; the current speaker pauses briefly: "One moment while we reshare the slides." |
| Audience reports no audio | Chat monitor messages the speaker; the speaker checks mute and repeats the last sentence. |
| Running over time | Shorten slides 13 and 16 to their first sentence each; never cut slide 15 (results) or 17 (conclusion). |

---

## Part 1 — Quan — Slides 1–6 (≈ 6 min)

### Slide 1 — Title *(≈ 20 s)*

*[Screen is shared and showing the title slide before you start talking.]*

Good morning, everyone. You should now see our title slide; if you cannot see the slides or hear me, please tell us in the chat.

We are Group 4: I am Minh-Quan Pham Nguyen, and with me are Tinh Nguyen Van and Thien-Lac Quach. Our topic today is LoRA and O-LoRA, two methods for fine-tuning large models while training only a very small number of parameters. Please keep questions in the chat during the talk; we will answer them at the end.

**▶ NEXT**

### Slide 2 — Guiding question and outline *(≈ 45 s)*

Our talk is built around one question, shown at the top of the slide: *how much does LoRA reduce trainable parameters, and what evidence do we need to judge its quality and resource use?*

The question has two halves on purpose. The first half, counting parameters, is simple arithmetic. The second half is harder, because fewer parameters does not automatically mean less memory, less time, or the same accuracy. We need to measure those.

Here is the plan, in the numbered list. I will cover the background and the LoRA formulation. Tinh will then explain training, merging, and a worked example on a 1024-by-1024 matrix. Lac will present O-LoRA, our experiment comparing LoRA with full fine-tuning on AG News, and the trade-offs.

**▶ NEXT**

### Slide 3 — Ways to adapt a pretrained model *(≈ 1 min)*

Suppose we already have a pretrained model and want it to do a new task. There are several ways to adapt it, and they differ in *what* we train. Please look at the table under the figure; I will go through it row by row.

*[Driver: move the cursor down the table row by row.]*

- **First row, linear probing,** freezes the whole backbone and trains only a new output head on top of its features. It is cheap, but the features themselves never change.
- **Second row, full fine-tuning,** is the other extreme: every weight gets updated, so we learn a full-size change ΔW for every matrix.
- **Third row, LoRA,** our focus, keeps the original weight W₀ frozen and adds a small trainable product A times B beside it.
- **Fourth row, adapters,** insert a small bottleneck network after a layer.
- **Last row, prompt and prefix tuning,** leave the weights alone and instead learn extra vectors that are prepended to the input or to the attention keys and values.

So LoRA sits between linear probing and full fine-tuning: it does change how each layer computes, but through a very small number of new parameters.

**▶ NEXT**

### Slide 4 — Intrinsic dimension of an objective *(≈ 1.5 min)*

Why should such a small update be enough? The idea comes from *intrinsic dimension*, studied by Li and colleagues in 2018. There are two panels on this slide, left and right.

*[Driver: cursor on the left panel.]* The left panel, labeled (a), is a toy parameter space with three parameters, so D equals 3. Training moves from the starting point θ₀ to a solution θ-star. Now suppose the set of good solutions is not a single point but a whole line, which we call S, with dimension s equal to 1. To reach that line we do not need to control all three directions; we only need to fix the directions that cross it. That number is the intrinsic dimension: D equals d-int plus s, so here d-int equals 2.

*[Driver: cursor on the right panel.]* The right panel, labeled (b), shows how it is measured. We fix the starting point θ₀ and a random matrix P, and we only optimize a small vector θ of dimension d. Every candidate solution is θ₀ plus P times that vector, so we search only inside the grey plane. The line S passes through this plane, so the plane, with d equal to 2, still reaches a solution. The problem can be solved in that small subspace.

Aghajanyan and colleagues applied this to pretrained language models and found that fine-tuning them needs a surprisingly small d. That finding is the motivation for LoRA.

**▶ NEXT**

### Slide 5 — Why a low-rank update? *(≈ 1 min)*

Let us connect this to LoRA precisely.

First, low-rank structure appears often in learning problems and trained networks. Second, fine-tuning has low intrinsic dimension: a small random subspace is enough.

But — third bullet — we want to be careful here. Intrinsic dimension counts free coordinates for the *whole* parameter vector. Rank is a property of *each individual matrix* ΔW. These are different notions. So low intrinsic dimension *motivates* the LoRA idea, but it does not prove it.

That is why LoRA states it as a hypothesis, in bold: the weight change a task needs has rank much smaller than the matrix dimensions. *If* that hypothesis holds, the consequence is direct: freeze W₀, and learn ΔW only in factored form, as A times B.

**▶ NEXT**

### Slide 6 — LoRA formulation *(≈ 1.5 min)*

Here is the whole method in one line, the large formula at the top: h equals W₀x plus α over r times A B x.

W₀ is the frozen d-by-k pretrained matrix. A is d-by-r, B is r-by-k, and the rank r is much smaller than both d and k. So the product AB has the same shape as W₀, but its rank is at most r.

The four bullets below make four points:

1. **Objective.** We still minimize the usual negative log-likelihood of the task data, but the variables we optimize, Θ, are only the LoRA factors — far fewer than the original parameters Φ₀.
2. **Rank and scaling.** r sets how much capacity the update has. The scaling α over r means that when we change r, we do not have to retune the learning rate as much.
3. **Initialization.** A starts Gaussian and B starts at zero, so AB equals zero. At step zero the model is *exactly* the pretrained model; training only moves it away gradually.
4. **Cost.** For each adapted matrix we train r times (d plus k) numbers instead of d times k. Gradients and optimizer states shrink in the same way.

That is the formulation. Tinh, over to you for training and deployment.

*[Quan mutes. Tinh unmutes. Quan stays the driver.]*

**▶ NEXT**

---

## Part 2 — Tinh — Slides 7–10 (≈ 4.5 min)

### Slide 7 — LoRA training: only A and B receive gradients *(≈ 1 min)*

Thanks, Quan. The diagram on the left shows one adapted layer during training.

*[Driver: trace the two paths in the diagram as Tinh describes them.]*

The input x takes two paths. The first is the frozen layer, giving W₀x. The second is the adapter: B first compresses x from k dimensions down to r dimensions, then A expands it back up to d dimensions. That adapter output is multiplied by the scale s, which is α over r, and added to the frozen output.

During backpropagation the loss gradient flows through the whole network, but only A and B are updated. W₀ is frozen.

One thing to remember for later, the last bullet: frozen does not mean free. W₀ is still stored on the GPU and still used in every forward pass. This will matter when we talk about memory.

**▶ NEXT**

### Slide 8 — After training: separate adapter or merged weights *(≈ 1 min)*

After training we have two deployment choices, shown side by side.

*[Driver: cursor on the left figure.]* On the left, we keep the adapter separate. The model still computes two paths, W₀x plus sABx. The advantage is that one shared base model can serve many tasks, each with its own small adapter, and we can switch tasks by swapping adapters. The limitation is that an adapter only works with the exact base model it was trained on.

*[Driver: cursor on the right figure.]* On the right, we merge. Because both paths are linear in x, we can add the update directly into the weight: W-merged equals W₀ plus sAB. Now the layer is again a single matrix multiplication, exactly like the original. So a merged LoRA model has no extra latency from a LoRA branch.

**▶ NEXT**

### Slide 9 — Worked example: counting trainable parameters *(≈ 1.5 min)*

Now let us put numbers on this. Take one weight matrix of size 1024 by 1024, so d and k both equal 1024. We count only the weights — no biases, no classifier.

*[Driver: cursor on the left column.]*

Full fine-tuning updates every entry: 1024 times 1024, which is 1,048,576 — just over one million parameters.

LoRA trains A and B. Each has 1024 times r entries, so together they have 2048 times r.

Take r equal to 4. A and B have 4,096 parameters each, so 8,192 in total. Divide: one million and forty-eight thousand over 8,192 is 128. So LoRA trains 128 times fewer parameters — a 99.2 percent reduction.

*[Driver: cursor on the right column, then down the table.]* In the right column: for this square matrix the general rule is simple. The ratio is 512 over r, and the reduction is 1 minus r over 512. In the table, r equal to 8 gives 64 times fewer; r equal to 16 gives 32 times fewer.

The trade-off is clear: doubling r doubles the number of parameters. A smaller r is cheaper; a larger r gives the update more capacity.

**▶ NEXT**

### Slide 10 — Trainable parameters ≠ GPU memory *(≈ 1 min)*

But a 128-times cut in trainable parameters is *not* a 128-times cut in GPU memory.

*[Driver: cursor on the left table.]* The left table lists what training has to hold. LoRA removes the gradients and the Adam optimizer states for W₀. But it still keeps W₀ itself, and it still needs activations and temporary tensors.

*[Driver: cursor on the right table.]* The right table counts bytes for the same matrix, r equal to 4, in 32-bit floats with Adam. Full fine-tuning needs 4 mebibytes for the weights, 4 for the gradients, and 8 for Adam's two moment estimates — 16 mebibytes in total. LoRA still needs the 4 mebibytes of frozen weights, plus 32 kibibytes for A and B, 32 for their gradients and 64 for Adam — about 4.13 mebibytes.

So trainable parameters dropped 128 times, but this memory dropped only about 3.9 times, because the frozen W₀ dominates. And once we add activations, which both methods need, the gap narrows further. Keep this in mind when Lac shows our measured memory.

Lac, over to you.

*[Tinh mutes. Lac unmutes.]*

**▶ NEXT**

---

## Part 3 — Lac — Slides 11–18 (≈ 7.5 min)

### Slide 11 — O-LoRA: orthogonality and the core hypothesis *(≈ 1 min)*

Thanks, Tinh. So far we adapted a model to one task. Now, as the top line of the slide says, suppose tasks arrive one after another: task 1, then task 2, and so on, and when we learn task t we only have task t's data. Learning a new task can then overwrite what the model learned before. This is called *catastrophic forgetting*. O-LoRA, by Wang and colleagues in 2023, targets exactly this setting.

*[Driver: cursor on the left column.]* In the left column, the key tool is orthogonality. Two subspaces are orthogonal if every vector in one is perpendicular to every vector in the other. For the column spaces of two LoRA matrices, A-i and A-t, we only need to check their columns, which means checking that A-i-transpose times A-t equals zero. Each entry of that small r-by-r matrix is the inner product of one column from each, so its squared Frobenius norm measures how much the two subspaces overlap. It is zero only when they are orthogonal.

*[Driver: cursor on the right column.]* In the right column is the hypothesis. The LoRA update, A-t B-t x, always lies in the column space of A-t, so that column space approximates the subspace where task t changes the model. If each new task updates in a direction orthogonal to earlier tasks, it should interfere less with them. So: one LoRA per task, with subspaces kept approximately orthogonal.

**▶ NEXT**

### Slide 12 — O-LoRA: method *(≈ 1 min)*

Here is how it is done.

First bullet: each task gets its own LoRA. When we train task t, the base weight W₀ and all earlier factors stay frozen; only A-t and B-t are trained. After t tasks, the effective weight is W₀ plus the sum of all t scaled updates.

Second bullet, the loss: the usual task loss, plus λ times the sum, over every earlier task, of the squared Frobenius norm of A-i-transpose A-t. This pushes the new subspace away from all old ones.

Two practical advantages: O-LoRA needs no replay data from old tasks, and because the updates can be merged into the weights, the model does not need to know which task an input comes from.

Two honest limitations, in the last bullet: a finite λ *encourages* orthogonality but does not guarantee it, so forgetting is reduced but not eliminated. And the stored adapters grow with the number of tasks.

**▶ NEXT**

### Slide 13 — LoRA and O-LoRA compared *(≈ 1 min)*

This table compares the two methods: LoRA in the middle column, O-LoRA on the right.

*[Driver: cursor down the rows.]*

Row by row: LoRA's goal is to adapt with few trainable parameters; O-LoRA's goal is to learn tasks in sequence with limited interference. LoRA trains one pair A, B; O-LoRA trains only the current pair and freezes the earlier ones. LoRA uses only the task loss; O-LoRA adds the orthogonality penalty. LoRA has no explicit mechanism against forgetting; O-LoRA aims to reduce it but cannot guarantee it. And O-LoRA's storage grows as tasks accumulate.

One caution, in the line below the table: the two papers use different models, data and settings, so this is a conceptual comparison, not a ranking of which performs better.

**▶ NEXT**

### Slide 14 — Experiment setup *(≈ 1 min)*

Now to our own experiment. We fine-tuned DistilRoBERTa on AG News, a news classification dataset with four classes: World, Sports, Business and Sci/Tech.

We held out a stratified 10 percent of the training data for validation, with seed 42, and used the official 7,600-example test split only once, at the very end. Inputs were capped at 256 tokens. Both runs used mixed precision on one GPU, three epochs, an effective batch of 32 — 8 per step times 4 accumulation steps — AdamW, and a linear schedule with warmup. We kept the checkpoint with the best validation macro F1.

*[Driver: cursor on the table.]* The table at the bottom shows what differs. The LoRA run adapts only the query and value matrices in attention, plus the classifier head. That is 0.8944 percent of all parameters — less than one percent. Full fine-tuning trains everything. The learning rates differ — two times ten to the minus four for LoRA, two times ten to the minus five for full fine-tuning — because LoRA usually needs a larger one. LoRA used rank 8, α 16, dropout 0.05.

**▶ NEXT**

### Slide 15 — Results on the AG News test split *(≈ 1.5 min)*

*[Driver: cursor on the left table.]* The left table compares the two methods. Full fine-tuning reached 95.03 percent accuracy and macro F1. LoRA reached 93.76 percent — 1.26 percentage points lower — while training under 1 percent of the parameters.

On resources, LoRA trained in 47 minutes versus 66, about 19 minutes faster. Peak allocated GPU memory was 1.10 gibibytes versus 2.14 — roughly half.

This is the point Tinh prepared us for: we cut trained parameters by more than a hundred times, but memory only halved. The frozen model, activations and other overhead are still there.

*[Driver: cursor on the right table.]* The right table shows LoRA per class. Sports is easiest, with F1 near 0.99. Business is hardest, at about 0.90. Out of 7,600 test examples there are 474 errors, and the largest confusion is between Business and Sci/Tech — 139 Business articles predicted as Sci/Tech, and 108 the other way. That makes sense: many news stories about technology companies are both.

**▶ NEXT**

### Slide 16 — Limitations and when LoRA may not fit *(≈ 1 min)*

We want to be clear about what this evidence can and cannot show.

*[Driver: cursor on the left column.]* On the left, the limits of our evidence. We have one run per method, so no variance estimate. The two methods used different learning rates, so the gap is not due to the method alone. We did not compare ranks. The evaluation times of about 30 seconds each are not a controlled latency test, and we did not time merged versus unmerged adapters. The 1024-by-1024 example is one matrix, not a whole model. And we did not run O-LoRA.

*[Driver: cursor on the right column.]* On the right, when LoRA may not be the best choice. When the task needs broad changes in the model's behavior. When the chosen rank cannot reach the quality target. For small models with plenty of compute, where full fine-tuning is simply easier. And deployment matters: separate adapters make switching tasks easy, while merged weights make serving one model simple.

**▶ NEXT**

### Slide 17 — Conclusion *(≈ 1 min)*

To answer our guiding question:

- **Parameters:** for one 1024-by-1024 matrix, rank-4 LoRA trains 8,192 weights instead of over a million — 99.22 percent fewer.
- **Quality:** training only 0.8944 percent of DistilRoBERTa's parameters, LoRA reached 93.76 percent accuracy on AG News, 1.26 points below full fine-tuning.
- **Resources:** LoRA was faster and used about half the peak memory. The savings are real, but much smaller than the parameter ratio suggests.
- **O-LoRA** extends LoRA to sequential tasks by penalizing overlap between task subspaces.
- **Next steps:** repeat the runs, match the learning rates, sweep the rank, and measure merged versus unmerged latency.

In one sentence, as at the bottom of the slide: LoRA fits when limiting trained parameters matters and the measured quality trade-off is acceptable.

**▶ NEXT**

### Slide 18 — References *(≈ 20 s)*

These are our main sources: the original LoRA and O-LoRA papers, the two intrinsic-dimension papers, the work on low-rank structure, and the AG News dataset paper.

Thank you for listening. We are happy to take your questions now — you can unmute and ask, or type in the chat and we will read the question aloud.

*[Quan and Tinh unmute for Q&A. Driver stays on slide 18, or jumps back to a slide when a question refers to it.]*

---

## Q&A — online handling

- **Read chat questions aloud** before answering, so everyone (and any recording) hears the question.
- **Who answers.** By default, the person who presented the relevant slides: Quan for slides 1–6, Tinh for 7–10, Lac for 11–18. The chat monitor names the answerer: "Tinh, this one is about the memory slide."
- **One voice at a time.** Others stay muted until the answerer finishes, then may add a short follow-up.
- **Showing a slide.** The answerer asks the driver: "Could we go back to slide 15?"
- **Prepared answers.** See the "Likely questions" appendix in `presentation_explained.md`.
- **Closing.** When time is up, Quan says: "Thank you all again. Any remaining questions can go in the chat and we will reply there," then stops sharing.
