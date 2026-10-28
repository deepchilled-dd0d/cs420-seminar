# CS420 seminar - LoRA exploration

## Required tasks

1. Distinguish pretraining, full fine-tuning, linear probing, adapters, and
   parameter-efficient fine-tuning (PEFT).
2. Explain low-rank factorization, rank r, and scaling factor α. Identify which
   parameters are frozen and which are trained.
3. Calculate trainable-parameter counts for a 1024 × 1024 weight matrix using
   full finetuning and LoRA ranks 4, 8, and 16. Estimate the reduction ratios,
   and distinguish parameter counts from actual GPU memory consumption.
4. Illustrate LoRA training and inference pipelines, including adapter merging.
   Explain intuitively how QLoRA differs from LoRA.
5. If resources permit, fine-tune a very small model or synthetic layer and
   compare two ranks. Otherwise, use a toy linear-layer experiment and compare
   parameter counts with a simple quality measure.
6. Analyze trade-offs among model quality, memory use, training time, inference
   efficiency, and experimental risks. Describe when LoRA may not be the best
   choice.
