# Presentation Q&A: LoRA and O-LoRA

Ten likely audience questions, grouped by topic, based on [report.tex](report/report.tex), its included sections, and [presentation.tex](report/presentation.tex).

## 1. Motivation and fundamentals

### Q1. Why should a low-rank update be enough to adapt a pretrained model?

**Answer:** The pretrained model already contains useful representations, so a downstream task may need changes in only a few directions. LoRA models this with a low-rank weight update. Low intrinsic dimension motivates this hypothesis, but does not prove low matrix rank or guarantee success on every task.

### Q2. How does LoRA differ from linear probing and full fine-tuning?

**Answer:** Linear probing trains only a classification head; full fine-tuning updates the entire model. LoRA freezes the pretrained weights and trains low-rank factors inside selected layers, allowing internal representations to adapt with fewer trainable parameters. Our experiment also trains the classifier.

## 2. LoRA formulation and parameter savings

### Q3. What do rank and scaling control, and why initialize one factor to zero?

**Answer:** In `h = W₀x + (α/r)ABx`, rank `r` limits the update's capacity, while `α/r` scales its contribution. Following the slides' notation, initializing `A` randomly and `B` to zero makes the initial update zero, preserving the pretrained layer's output. Larger ranks increase capacity and parameter cost; we did not test their effect on accuracy.

### Q4. How do we obtain the claimed 99.22% reduction in trainable parameters?

**Answer:** A `d × k` matrix has `dk` weights, while its LoRA factors have `r(d + k)` parameters. For a `1024 × 1024` matrix at rank 4, this is 8,192 instead of 1,048,576: 128 times fewer, or a 99.21875% reduction. This example covers one matrix; our experiment trains 0.8944% of the model's parameters, including the classifier.

## 3. Memory and deployment

### Q5. Why does a huge parameter reduction produce only about half the GPU memory use?

**Answer:** LoRA reduces gradients and optimizer states, but still stores the frozen base weights and needs activations and temporary tensors. In our runs, peak allocated memory was 1.099 GiB for LoRA versus 2.138 GiB for full fine-tuning. Total memory therefore does not shrink in proportion to trainable parameters.

### Q6. Does LoRA add inference latency, and should we merge the adapter?

**Answer:** Keeping the adapter separate adds a computation branch and makes task switching convenient. Merging it as `W_merged = W₀ + (α/r)AB` removes that branch and preserves the original layer's matrix-multiplication structure. We did not measure merged versus unmerged latency, so our evaluation runtimes cannot establish a deployment speedup.

## 4. O-LoRA and continual learning

### Q7. How does O-LoRA try to reduce catastrophic forgetting?

**Answer:** O-LoRA learns a new adapter for each sequential task while freezing the base and earlier adapters. It adds the penalty `λ Σᵢ<t ‖AᵢᵀAₜ‖²_F` to discourage overlap between current and earlier update subspaces. The aim is to reduce interference without replaying earlier tasks' data.

### Q8. Does orthogonality guarantee no forgetting, and what does O-LoRA cost?

**Answer:** No. A finite penalty encourages approximate orthogonality, and subspace separation does not guarantee unchanged predictions on earlier tasks. Retained adapter storage grows with the number of tasks, although accumulated updates can be merged for inference without selecting a task ID. We reviewed O-LoRA conceptually and did not evaluate it experimentally.

## 5. Experimental evidence and limitations

### Q9. What did the AG News experiment show, and when would we choose LoRA?

**Answer:** LoRA reached 93.76% accuracy versus 95.03% for full fine-tuning, about 1.26 percentage points lower. It trained in 47.31 versus 66.38 minutes and used less peak allocated memory. LoRA is attractive when these resource savings justify the measured quality loss; full fine-tuning may suit a stricter quality target when resources permit.

### Q10. How strong is the comparison, and what experiments should come next?

**Answer:** These are observations from one run per method, with different learning rates and no rank sweep or variance estimate. The notebook specifies rank 8, but the saved LoRA result files do not record it. Next steps are repeated seeds, controlled learning-rate comparisons, rank sweeps with complete metadata, merged/unmerged latency measurements, and a sequential-task evaluation of O-LoRA.
