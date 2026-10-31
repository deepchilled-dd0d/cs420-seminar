# Task divison
**Writer**: Lạc  
**Reviewers**: Quân, Tĩnh  
**Version**: 1.0.2 
**Last edited**: Oct 31 2026  

--- 
## Rules:
- Each task will have a deliverable of some kind (except for presentation). Task completion means the deliverable is present on Github or communicated to other members.
- Each task will have a deadline. Try to commit to it, or tell the remaining members if there's any problem.
- Use kebab-case for documents (except paper names) and snake_case for code (here I assume code will be Python).
- To ensure a cleaner workflow, for each task (or many related tasks) make a branch with naming format `[Task 1 ID]_[Task 2 ID]_...`. Open PR when you are done.
- For the same reason as above, PDF files generated from out LaTeX files will not be pushed to Github; each member compiles his own local version of the files. I will configure a GitHub Action to compile PDF on Github to ensure consistent version. 

## Phases:
From my perspective, this seminar consists of three phases: research, writing, and presentation. Assuming a pessimistic time budget from today until Nov 12, I will list the tasks required for each phase of the seminar and divide them among us.

### Research
--- 
Expected window: 8 days (Oct 31 - Nov 7)  

---
|Task ID| Task Description | Asignee | Deliverable | Deadline | Note |
| --- |  --- | :---: | --- | :---: | --- |
|RE-01| Distinguish pretraining, full fine-tuning, linear probing, adapters, and parameter-efficient fine-tuning (PEFT). | Quân | `docs/assets/fundamental-diagram.png` | Nov 2 | Independent starting task. Formalize the hand-drawn diagram for WR-02 and the slides. |
|RE-02| Explain low-rank factorization, rank \(r\), and scaling factor \(\alpha\). Identify which parameters are frozen and which are trained. | Lạc | `docs/summary/LoRA-summary.tex` and `docs/summary/CL-LoRA-summary.tex` | Nov 3 | Foundational task for RE-03, RE-04, RE-05, and the method section; share the LoRA formula as soon as it is ready. |
|RE-03| Calculate trainable-parameter counts for a \(1024 x 1024\) weight matrix using full fine-tuning and LoRA ranks 4, 8, and 16. Estimate the reduction ratios, and distinguish parameter counts from actual GPU memory consumption. | Tĩnh | `docs/assets/calculation.md` | Nov 4 | Depends on the dimensions and formula in RE-02; supplies the worked example for WR-04. |
|RE-04| Illustrate LoRA training and inference pipelines, including adapter merging; explain intuitively how QLoRA differs from LoRA (a frozen quantized base model plus low-rank adapters). | Quân | `docs/assets/LoRA-pipeline.png` and `docs/summary/adapter-methods-comparison.tex` | Nov 5 | Depends on RE-02. Draw the LoRA pipeline first; Update: CL paper is allowed so compare those papers against LoRA. |
| RE-05 | If resources permit, fine-tune a very small model or synthetic layer and compare two ranks. Otherwise, use a toy linear-layer experiment and compare parameter counts with a simple quality measure. | Lạc | Kaggle notebook in `src/` and results in JSON format stored in `docs/test-results/` | Nov 6 | Start after RE-02 fixes the method and compared ranks; results feed WR-05. Existing N-LoRA code is available if useful. |
|RE-06| Analyze trade-offs among model quality, memory use, training time, inference efficiency, and experimental risks. Describe when LoRA may not be the best choice. | Tĩnh | `docs/summary/trade-off.tex` | Nov 7 | Synthesize RE-03 to RE-05, including experimental limitations, for WR-05 and WR-06. |

### Writing
--- 
Expected window: 4 days (Nov 8 - Nov 11)  

---
| Task ID | Task Description | Assignee | Deliverable (all tasks share folder `docs/report/`; I only name the file name below) | Deadline | Note |
| --- | --- | :---: | --- | :---: | --- |
| WR-01 | Write the Abstract. | Tĩnh | `abstract.tex` | Nov 10 | Write after WR-02 to WR-06 have drafts so it summarizes the actual report. |
| WR-02 | Write the Introduction. | Quân | `introduction.tex` | Nov 8 | Draw on RE-01 and RE-02; establishes the question and scope for the report. |
| WR-03 | Write the Background/Paper Review. | Quân | `background.tex` | Nov 8 | Draw on RE-02 and RE-04; distinguish the required LoRA paper from the QLoRA comparison. |
| WR-04 | Write the Method/Worked Example. | Lạc | `method.tex` | Nov 9 | Depends on RE-02, RE-03, and RE-04 for the equation, parameter table, and pipeline. |
| WR-05 | Write the Results and Discussion. | Tĩnh | `results.tex` | Nov 9 | Depends on RE-05 results and RE-06 trade-offs; identify team results separately from paper findings. |
| WR-06 | Write the Conclusion. | Tĩnh | `conclusion.tex` | Nov 10 | Write after WR-04 and WR-05 so the claims match the method and evidence. |
| WR-07 | Compile and format the References. | Lạc | `report.pdf` and bibliography | Nov 11 | Final compile follows WR-01 to WR-06; check references and formatting after all sections are in place. |
| WR-08 | Make the presentation. | Lạc | `presentation.pdf` | Nov 10 | Build from the research assets and draft report sections so WR-09 has a full day to review. |
| WR-09 | Review the presentation. | Quân, Tĩnh | Feedback if any | Nov 11 | Depends on WR-08; check accuracy, slide coverage, citations, and the 15-minute speaking plan. |

### Presentation
---
Time window: the presentation day (assume Nov 12 until further notice)

---

Plan for 10 content slides in 15 minutes, followed by approximately 5 minutes of Q&A. The title and references slides are separate. Cite figures, tables, equations, and research findings on the relevant slides.

| Task ID | Task Description | Assignee | Note |
| --- | --- | :---: | --- |
| PR-01 | Slide 1 — Introduce the guiding question, objectives, and scope. | Quân | 1 minute. Explain why adapting a large model without updating all its weights matters. |
| PR-02 | Slide 2 — Compare pretraining, full fine-tuning, linear probing, adapters, and PEFT. | Quân | 1 minute. Use the diagram from RE-01. |
| PR-03 | Slide 3 — Explain LoRA with a diagram of frozen base weights and trainable low-rank matrices. | Quân | 2 minutes. Show \(W + BA\) and distinguish the paper's method from the team's illustration. |
| PR-04 | Slide 4 — Define \(W\), \(A\), \(B\), \(r\), and \(\alpha\); explain \(W' = W + (\alpha/r)BA\) and matrix dimensions. | Lạc | 2 minutes. State assumptions and identify the trained parameters. |
| PR-05 | Slide 5 — Work through trainable-parameter counts for a 1024 × 1024 matrix using full fine-tuning and LoRA ranks 4, 8, and 16. | Lạc | 2 minutes. Show reduction ratios from RE-03 and distinguish parameter counts from total GPU memory. |
| PR-06 | Slide 6 — Explain LoRA training, inference, and adapter merging. | Lạc | 1 minute. Use the pipeline diagram from RE-04. |
| PR-07 | Slide 7 — Explain QLoRA's frozen quantized base model and trained low-rank adapters. | Lạc | 1 minute. State the difference from standard LoRA. |
| PR-08 | Slide 8 — Present the toy experiment or small-model results, including baseline, ranks, quality measure, and limitations. | Lạc | 2 minutes. If results are unavailable, show the annotated workflow and worked numerical example. |
| PR-09 | Slide 9 — Compare model quality, memory use, training time, and inference efficiency. | Tĩnh | 1.5 minutes. Separate paper findings from the team's results and interpretation. |
| PR-10 | Slide 10 — Summarize when LoRA is useful, when it may be a poor fit, and the answer to the guiding question. | Tĩnh | 1.5 minutes. Keep the conclusion tied to the evidence shown. |
| PR-11 | Rehearse transitions and keep the content portion within 15 minutes. | Quân, Lạc, Tĩnh | Speaking split: Quân 4 minutes, Lạc 8 minutes, Tĩnh 3 minutes. |
| PR-12 | Prepare for and answer TA questions about the method, calculations, QLoRA, and results. | Quân, Lạc, Tĩnh | Each member answers at least one question during the approximately 5-minute Q&A. |

