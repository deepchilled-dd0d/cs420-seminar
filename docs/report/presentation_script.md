# Presentation Script — LoRA and O-LoRA: Parameter-Efficient Fine-Tuning of Large Models

CS420 Seminar, Group 4 — 23 November 2026 — **online presentation**

Slide numbers follow the compiled deck (`presentation.tex`, 14 frames). Target length is about 16 minutes, at roughly 125 spoken words per minute (a little slower than in person, because of audio lag and compression).

| Part | Speaker | Slides | Topic | Suggested time |
|------|---------|--------|-------|----------------|
| 1 | Quan | 1–5 | Motivation, background, LoRA formulation | ≈ 5.5 min |
| 2 | Tinh | 6–9 | LoRA training, merging, worked examples | ≈ 4.5 min |
| 3 | Lac | 10–14 | O-LoRA, experiment, limitations, spoken wrap-up | ≈ 6 min |

**How to use this script**

- The slides already show the formulas, tables and numbers, and the audience can read them. The script **does not read the slides aloud**. Each slide's speech points to what is on screen in a few words, then spends its time on what the slide does *not* say: why it holds, what it means, and how it links to other slides.
- Plain text is spoken. *[Italic brackets]* are stage directions. **▶ NEXT** marks the moment the slide driver advances.

> Note: the speaker comments in `presentation.tex` still use the old assignment (Quan 2–5, Lac 6–11, Tinh 12–14). This script follows the current order: Quan, then Tinh, then Lac.

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
- **Asking for the next slide.** Speakers do not need to say "next slide"; the driver follows the **▶ NEXT** markers. If out of sync, the speaker says "Next slide, please" or "Could we go back to slide 8?".
- **Handovers.** The outgoing speaker names the next speaker and mutes. The incoming speaker unmutes **before** the handover line ends, then starts with "Thanks, …" — no "Can you hear me?" (the chat monitor will flag audio problems).

### If something goes wrong

| Problem | What to do |
|---|---|
| Speaker's connection drops | The next speaker in order continues from the current slide using this script. The dropped speaker rejoins and stays muted. |
| Driver's connection drops | Lac starts sharing the PDF from the current slide; the current speaker pauses briefly: "One moment while we reshare the slides." |
| Audience reports no audio | Chat monitor messages the speaker; the speaker checks mute and repeats the last sentence. |
| Running over time | Drop the "Beyond the slide" examples marked *(optional)*; never cut slide 13 (results) or the wrap-up's last sentence. |

---

## Part 1 — Quan — Slides 1–5 (≈ 5.5 min)

### Slide 1 — Title *(≈ 20 s)*

*[Screen is shared and showing the title slide before you start talking.]*

Good morning, everyone. You should now see our title slide; if you cannot see the slides or hear me, please tell us in the chat.

We are Group 4: I am Minh-Quan Pham Nguyen, and with me are Tinh Nguyen Van and Thien-Lac Quach. Our topic is LoRA and O-LoRA: how to adapt a large pretrained model while training only a tiny fraction of its weights. Please keep questions in the chat; we will answer them at the end.

**▶ NEXT**

### Slide 2 — Outline *(≈ 45 s)*

The question at the top is the thread of the whole talk. LoRA is usually advertised with one number: how many fewer parameters it trains. But what people actually pay for is GPU memory, training time and accuracy, and the parameter count does not automatically tell you any of those. So we will first see where that number comes from, and then check, in our own experiment, how much of it survives as real savings.

The six items on screen follow that thread: why low-rank updates should work, how LoRA works, a parameter count, an extension to several tasks in a row, our experiment, and its limits. I will cover the background and the formulation, Tinh the training and the worked examples, and Lac O-LoRA and our experiment.

**▶ NEXT**

### Slide 3 — Pretrained model adaptation *(≈ 1 min)*

The figure is the usual pipeline: pretrain once on broad data, then adapt to each task. The table lists five ways to adapt. Rather than read the formulas, let me give two questions that separate them.

First: *what do we train?* Linear probing trains only a new head, so if the pretrained features do not fit the task, nothing can fix them. Full fine-tuning trains everything, which is powerful but means a full copy of the model for every task.

Second: *does the method cost anything at inference?* Adapters put a small network with a nonlinearity, the σ, into the forward path, so they always add latency. Prompt and prefix tuning add tokens, which use up sequence length. LoRA's update is linear — just a matrix product — and Tinh will show on slide 7 that this lets us fold it back into the original weight for free.

So LoRA aims for the middle: it changes every adapted layer, like fine-tuning, but trains very few numbers and costs nothing extra once merged.

**▶ NEXT**

### Slide 4 — Intrinsic dimension of an objective *(≈ 1.5 min)*

Why should so few trained numbers be enough? This slide gives the intuition, from Li and colleagues.

*[Driver: cursor on the left panel.]* The left panel says the solutions do not form one point but a whole set, here a line. That is what happens in heavily over-parameterized networks: many different weight settings work equally well. Moving *along* the solution set changes nothing, so only the directions that *cross* it matter. That count is the intrinsic dimension.

*[Driver: cursor on the right panel.]* The right panel is how it is measured. The key word is *random*: P is not chosen cleverly. In practice you increase the small dimension d until the model reaches, say, 90 percent of full-training performance. If even a random low-dimensional subspace gets you there, the problem really only needs a few directions.

Aghajanyan and colleagues did this for pretrained language models. In one of their results, a RoBERTa model reached 90 percent of full fine-tuning performance on a paraphrase task by optimizing only about 200 numbers. They also found that bigger and better-pretrained models need *fewer* — which is exactly the regime where LoRA is used.

**▶ NEXT**

### Slide 5 — Why a low-rank update, and LoRA formulation *(≈ 2 min)*

*[Driver: cursor on the left column.]* The left column gives two pieces of evidence: networks show low-rank structure, and fine-tuning has low intrinsic dimension. The third bullet is the subtle part, so let me explain it.

Intrinsic dimension is about the *whole* parameter vector, searched through one random projection. Rank is a property of *each individual matrix*. And the two really are different: in the random-subspace experiment, the update to each weight matrix is a random mixture of directions, so it is typically *full* rank — even though it lives in a tiny subspace overall. So low intrinsic dimension does not give us low rank for free. That is why LoRA has to *assume* it, as a hypothesis, and then test it by whether the method works.

*[Driver: cursor on the right column.]* If we accept that hypothesis, any rank-r matrix can be written as A times B, and that is the formula on the right. Three things on this column are worth explaining.

Why is B set to zero at the start? Because then AB is zero and the model begins *exactly* as the pretrained model, so training cannot damage it on step one. Why not set both to zero? Then neither factor would ever get a gradient, and nothing would train.

Why divide by r? A larger r adds more terms to the product, so the update would grow just from changing r. The α over r keeps its size stable, so we can try different ranks without retuning the learning rate.

And the last bullet, r times d plus k instead of d times k, is the parameter saving Tinh will put numbers on. One footnote: some papers swap the names A and B; the math is the same.

Tinh, over to you.

*[Quan mutes. Tinh unmutes. Quan stays the driver.]*

**▶ NEXT**

---

## Part 2 — Tinh — Slides 6–9 (≈ 4.5 min)

### Slide 6 — LoRA training *(≈ 1 min)*

Thanks, Quan. The diagram shows one adapted layer, and the bullets describe its two paths. What the slide does not show is what this means for *compute*.

Freezing W₀ does not mean we skip it. To train A and B in earlier layers, the gradient still has to flow *backwards through* W₀, so that matrix multiplication still happens in both the forward and the backward pass. What LoRA actually skips is computing the gradient *for* W₀ itself, and the optimizer update for it.

So expect LoRA to train faster, but nowhere near "a hundred times faster" — most of the arithmetic is still there. Keep that in mind for our measured training time on slide 13.

One more detail not on the slide: in our experiment, a small dropout of 5 percent is applied only on the adapter path, as regularization.

**▶ NEXT**

### Slide 7 — LoRA after training *(≈ 1 min)*

Two deployment options are on screen. The reason the right-hand option, merging, is even possible is linearity: W₀x plus sABx equals W₀ plus sAB, all times x. You compute that sum once and the layer is back to its original shape and speed. Adapters cannot do this, because of the nonlinearity inside them.

The left-hand option matters because of storage. In the LoRA paper, the per-task checkpoint for GPT-3 shrank from about 350 gigabytes to about 35 megabytes. *(optional)* In our own run, the LoRA weights are only about half a megabyte, plus about 2 megabytes for the classifier head, against over 300 megabytes for the full model.

A practical point: once merged, switching tasks means subtracting the update again, or keeping a clean copy of W₀.

**▶ NEXT**

### Slide 8 — LoRA example: counting trainable parameters *(≈ 1.5 min)*

The arithmetic is on screen: for this 1024-by-1024 matrix, rank 4 trains 128 times fewer parameters. Three things it does not say.

First, the ratio is d over 2r for a square matrix, so it grows with the matrix size. The same rank 4 on a GPT-3-sized matrix, about 12 thousand wide, saves over 1,500 times. That is why LoRA is most attractive for the largest models.

Second, LoRA only saves anything while r is below 512 here — beyond that, the two factors are bigger than the matrix. In practice r is between about 4 and 64.

Third, connecting to our real model. *(optional)* DistilRoBERTa's attention matrices are 768 by 768; at rank 8 each one trains 48 times fewer parameters. All twelve query and value adapters together are about 147 thousand parameters. But the new classifier head is about 594 thousand — so most of what we actually train in our experiment is the head, not the adapters.

**▶ NEXT**

### Slide 9 — LoRA example: trainable parameters ≠ GPU memory *(≈ 1 min)*

The slide shows that for this one matrix the memory drops only about 3.9 times, not 128. Here is the rule of thumb behind it.

With Adam, full fine-tuning stores four numbers per weight: the weight, its gradient, and two optimizer states. LoRA removes the last three for the frozen weights but must keep the weight itself. So on the weights, the best LoRA can ever do is about a 4-times saving — no matter how small r is.

Then add activations, which are the same for both methods and grow with batch size and sequence length. They push the real saving even lower. The LoRA paper itself reports exactly this pattern for GPT-3: ten thousand times fewer trainable parameters, but only about three times less GPU memory.

So our prediction for the experiment: a real memory saving, but small. Lac, over to you.

*[Tinh mutes. Lac unmutes.]*

**▶ NEXT**

---

## Part 3 — Lac — Slides 10–14 (≈ 6 min)

### Slide 10 — O-LoRA: orthogonality and core hypothesis *(≈ 1 min)*

Thanks, Tinh. The top line describes the problem: tasks arrive one after another. To make forgetting concrete: if task 2 is trained by changing the same directions in the weights that task 1 relied on, task 1's behavior is simply overwritten.

The left column gives the test for whether two subspaces overlap. The reason it uses A, and not B, is in the right column: the update A times Bx is always a combination of A's columns, whatever the input. So A's columns are the *output directions* a task is allowed to change. If task 2 only changes directions task 1 never used, it should interfere less.

Why the squared Frobenius norm? Because it adds up the overlap of every pair of columns into one smooth number that is zero only for perfect orthogonality — exactly what you want as a loss term.

There is a limit, too: a 768-dimensional layer only has room for 96 mutually orthogonal rank-8 subspaces.

**▶ NEXT**

### Slide 11 — O-LoRA: method *(≈ 1 min)*

The slide gives the loss: the task loss plus λ times the overlap with every earlier task. λ is a trade-off: larger λ forces more orthogonality but leaves less freedom to fit the new task. Notice the penalty only involves A, because A defines the directions; B is free.

Two properties are worth contrasting with a simpler idea, keeping one separate adapter per task. Separate adapters never interfere, but at test time you must know which task an input belongs to, to pick the right adapter. O-LoRA sums everything into one weight, so it needs no task label — and it needs no stored old data either, because old tasks are "remembered" through their frozen A matrices.

The last bullet is the honest part: the penalty only *encourages* orthogonality, so forgetting is reduced, not eliminated, and we keep every task's factors around.

**▶ NEXT**

### Slide 12 — Experiment setup *(≈ 1 min)*

Now to our experiment. The slide lists the setup; let me explain the choices behind it.

We hold out a *stratified* validation split so it keeps the same class balance, and we touch the test split only once, so no decision is tuned on it. We pick checkpoints by macro F1, which weights every class equally; on this balanced data it is almost the same as accuracy. The batch of 8 with 4 accumulation steps simply gives an effective batch of 32 within limited GPU memory.

*[Driver: cursor on the table.]* In the table, the learning rates differ by ten times. That is standard: LoRA's factors start from zero and are few, so they need a larger step, while full fine-tuning uses a small one to avoid destroying the pretrained features. But it also means our comparison changes two things at once — we come back to this in the limitations.

And the 0.89 percent trained: as Tinh mentioned, most of that is the classifier head, not the adapters.

**▶ NEXT**

### Slide 13 — Results on the AG News test split *(≈ 1.5 min)*

*[Driver: cursor on the left table.]* The headline is on screen: full fine-tuning is 1.26 points more accurate. Another way to see that gap: full fine-tuning makes about 378 errors on the test set, LoRA makes 474 — roughly a quarter more mistakes. Whether that matters depends on the application.

The resource numbers match what Tinh predicted. Training is about 29 percent faster, not a hundred times, because the forward and backward passes through the frozen model still happen. Peak memory roughly halves, because the frozen weights and activations stay. The paper's "ten thousand times fewer parameters, three times less memory" pattern shows up even in our small model.

*[Driver: cursor on the right table.]* Per class, the Business–Sci/Tech confusion explains the two weakest numbers: Business has the lowest recall because its articles leak into Sci/Tech, and Sci/Tech has the lowest precision because it receives them. Those 247 examples are over half of all errors, and many are genuinely ambiguous — think of a news story about a tech company's earnings.

**▶ NEXT**

### Slide 14 — Limitations *(≈ 1.5 min, including wrap-up)*

*[Driver: cursor on the left column.]* The left column lists what our evidence cannot show. What would fix each point: three to five runs with different seeds to get an error bar on that 1.26-point gap; tuning the learning rate for each method separately; repeating with ranks 4, 8 and 16; and a proper latency test, timing single predictions with merged and unmerged adapters.

*[Driver: cursor on the right column.]* The right column is about when LoRA is the wrong tool. The common thread is the low-rank hypothesis: if a task needs the model to change broadly — say, a language or domain far from pretraining — a low-rank update may simply not be enough.

*[Stay on this slide for the wrap-up; there is no conclusion slide.]*

So, to answer our guiding question. LoRA cuts trainable parameters dramatically — 128 times for one matrix at rank 4. But in our experiment that became a 1.26-point accuracy cost, about 29 percent faster training and roughly half the peak memory: real savings, far smaller than the parameter count suggests. O-LoRA carries the idea to tasks learned in sequence.

In one sentence: LoRA fits when limiting trained parameters matters and the measured quality trade-off is acceptable.

Thank you for listening. We are happy to take your questions — you can unmute and ask, or type in the chat and we will read the question aloud.

*[Quan and Tinh unmute for Q&A. Driver stays on slide 14, or jumps back to a slide when a question refers to it.]*

---

## Q&A — online handling

- **Read chat questions aloud** before answering, so everyone (and any recording) hears the question.
- **Who answers.** By default, the person who presented the relevant slides: Quan for slides 1–5, Tinh for 6–9, Lac for 10–14. The chat monitor names the answerer: "Tinh, this one is about the memory slide."
- **One voice at a time.** Others stay muted until the answerer finishes, then may add a short follow-up.
- **Showing a slide.** The answerer asks the driver: "Could we go back to slide 13?"
- **Prepared answers.** See the "Likely questions" appendix in `presentation_explained.md`.
- **Closing.** When time is up, Quan says: "Thank you all again. Any remaining questions can go in the chat and we will reply there," then stops sharing.
