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
   Explain intuitively how OLoRA differs from LoRA.
5. If resources permit, fine-tune a very small model or synthetic layer and
   compare two ranks. Otherwise, use a toy linear-layer experiment and compare
   parameter counts with a simple quality measure.
6. Analyze trade-offs among model quality, memory use, training time, inference
   efficiency, and experimental risks. Describe when LoRA may not be the best
   choice.

I. BACKGROUND AND FUNDAMENTALS

[x] 1. Explain the distinction between:
       - Pretraining
       - Full fine-tuning
       - Linear probing
       - Adapter-based fine-tuning
       - Parameter-efficient fine-tuning (PEFT)

[ ] 2. Understand the original LoRA method:
       - Explain low-rank matrix factorization.
       - Explain the rank r.
       - Explain the scaling factor alpha.
       - Identify frozen and trainable parameters.
       - Explain why low-rank updates can approximate
         weight updates during fine-tuning.

[x] 3. Calculate trainable parameters:
       - Consider a 1024 x 1024 weight matrix.
       - Calculate parameters for full fine-tuning.
       - Calculate parameters for LoRA with ranks 4, 8, 16.
       - Calculate parameter reduction ratios.
       - Distinguish parameter count from actual GPU memory
         consumption.


II. LORA AND O-LORA

[ ] 4. Understand and illustrate the LoRA training pipeline:
       - Loading a pretrained model.
       - Freezing original model weights.
       - Adding low-rank adapters.
       - Forward and backward propagation.
       - Updating adapter parameters.
       - Saving adapters.
       - Merging adapters into original weights.

[ ] 5. Understand and illustrate the LoRA inference pipeline:
       - Inference with separate adapters.
       - Inference after adapter merging.
       - Explain the difference in computational overhead.

[x] 6. Understand O-LoRA:
       - Identify the problem of catastrophic forgetting.
       - Explain why sequentially learning new tasks can
         interfere with previously learned tasks.
       - Explain the intuition behind orthogonal subspaces.
       - Understand how O-LoRA uses LoRA parameters to
         approximate previous task update subspaces.
       - Explain how new task adapters are constrained to
         be orthogonal to previous task subspaces.
       - Understand why previous task adapters are frozen.
       - Explain how this reduces interference between tasks.

[x] 7. Compare LoRA and O-LoRA:
       - Main objectives.
       - Mathematical formulations.
       - Training procedures.
       - Parameter requirements.
       - Handling multiple tasks.
       - Catastrophic forgetting.
       - Generalization to unseen tasks.
       - Advantages and limitations of each method.


III. MATHEMATICAL FORMULATIONS

[x] 8. Explain the original LoRA formulation:

       W' = W + Delta W

       Delta W = (alpha/r) BA

       - Define every symbol.
       - Explain all matrix dimensions.
       - Explain the rank constraint.
       - Explain the scaling factor.
       - Explain which parameters are trainable.

[x] 9. Explain the O-LoRA formulation:

       Delta W_t = A_t B_t

       Orthogonality constraint:

       A_i^T A_t = 0, for i < t

       Orthogonality regularization:

       L_orth = || A_i^T A_t ||_F^2

       - Define every symbol.
       - Explain the subspace represented by A_t.
       - Explain the orthogonality condition.
       - Explain the regularization term.
       - Explain how previous adapters are preserved.
       - Explain how multiple task-specific updates can
         be merged into the original weights.

[x] 10. Provide complete worked numerical examples:
        - Calculate a LoRA parameter update.
        - Demonstrate how rank affects parameter count.
        - Construct two low-rank subspaces.
        - Demonstrate orthogonality between two subspaces.
        - Calculate an orthogonality loss.


IV. EXPERIMENTS AND COMPARISON

[x] 11. Implement an experiment if resources permit:
        - Use a very small pretrained model or synthetic model.
        - Otherwise, use a toy linear-layer experiment.
        - Implement standard LoRA.
        - Implement a simplified O-LoRA experiment.
        - Compare performance across at least two ranks.

[ ] 12. Design a continual-learning experiment:
        - Define at least two sequential learning tasks.
        - Train on the first task.
        - Train on the second task without replaying
          the first task's training data.
        - Measure performance on both tasks.
        - Compare standard sequential LoRA with O-LoRA.
        - Observe whether learning the second task affects
          performance on the first task.
        - Analyze catastrophic forgetting.

[x] 13. Analyze experimental results:
        - Trainable parameter counts.
        - Model quality.
        - Forgetting on previous tasks.
        - Orthogonality between task subspaces.
        - Training time.
        - Memory consumption.
        - Inference efficiency.

[ ] 14. Discuss limitations and trade-offs:
        - When is LoRA sufficient?
        - When is O-LoRA more appropriate?
        - What happens when the number of tasks increases?
        - What are the limitations of orthogonal subspaces?
        - How does parameter growth affect continual learning?
        - When might neither method be appropriate?


V. EXPECTED OUTPUTS

[ ] 15. Produce a LoRA architecture diagram showing W + BA.

[ ] 16. Produce LoRA training and inference pipeline diagrams.

[ ] 17. Produce an O-LoRA diagram illustrating multiple
       approximately orthogonal task subspaces.

[ ] 18. Produce a trainable-parameter comparison table
       for full fine-tuning and different LoRA ranks.

[ ] 19. Produce a LoRA vs O-LoRA comparison table.

[ ] 20. Include either:
       - Experimental results from a toy or small-model experiment;
       OR
       - An annotated mathematical example and workflow.

[ ] 21. Explain all figures, equations, tables and assumptions.
       Clearly distinguish results reported by research papers
       from experiments conducted by the team.


VI. RESEARCH PAPERS

[ ] 22. Read the original LoRA paper:

       Hu, E. J. et al. (2022).
       LoRA: Low-Rank Adaptation of Large Language Models.
       ICLR 2022.

       https://mlanthology.org/iclr/2022/hu2022iclr-lora/

       Reading focus:
       - Low-rank weight updates.
       - W + BA.
       - Rank and scaling.
       - Trainable parameter reduction.
       - Training and inference.

[ ] 23. Read the O-LoRA paper:

       Wang, X. et al. (2023).
       Orthogonal Subspace Learning for Language Model
       Continual Learning.
       Findings of EMNLP 2023.

       https://arxiv.org/abs/2310.14152

       Reading focus:
       - Continual learning.
       - Catastrophic forgetting.
       - Orthogonal low-rank subspaces.
       - Orthogonality regularization.
       - Sequential task learning.
       - Experimental evaluation.

[ ] 24. Identify the relationship between the papers:
       - What problem does LoRA solve?
       - What limitation motivates O-LoRA?
       - What does O-LoRA introduce?
       - What improvements does O-LoRA demonstrate?
       - What limitations remain?


VII. REPORT AND SEMINAR PREPARATION

[ ] 25. Write a 6–10-page research-style report in Vietnamese.

       Required structure:
       - Abstract
       - Introduction
       - Background / Paper Review
       - Method / Worked Example
       - Results and Discussion
       - Conclusion
       - References

[ ] 26. Prepare 8–12 content slides.

[ ] 27. Prepare a 15-minute seminar presentation.

[ ] 28. Prepare for approximately 5 minutes of Q&A.

[ ] 29. Ensure every team member:
       - Presents at least one part.
       - Understands the mathematical formulations.
       - Can explain at least one experiment or example.
       - Answers at least one question.

[ ] 30. Verify all citations, figures, equations, datasets
       and AI-assisted claims before submission.
