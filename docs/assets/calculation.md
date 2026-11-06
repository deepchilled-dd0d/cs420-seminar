# RE-03: LoRA Trainable Parameter Calculation

## 1. Objective

Calculate and compare the number of trainable parameters for a weight matrix of size 1024 × 1024 using:

- Full Fine-Tuning
- LoRA with rank r = 4
- LoRA with rank r = 8
- LoRA with rank r = 16

Also, analyze parameter reduction ratios and distinguish trainable parameter counts from actual GPU memory consumption.

## 2. Given Dimensions

Consider an original weight matrix:

$$
W \in \mathbb{R}^{1024 \times 1024}
$$

For LoRA, the weight update is represented by two low-rank matrices:

$$
A \in \mathbb{R}^{r \times 1024}
$$

$$
B \in \mathbb{R}^{1024 \times r}
$$

The original weight matrix W is frozen, while A and B are trainable.

## 3. Full Fine-Tuning Calculation

Full Fine-Tuning updates every parameter in the original weight matrix.

$$
N_{\text{Full}} = 1024 \times 1024
$$

$$
\boxed{N_{\text{Full}} = 1,048,576}
$$

Therefore, Full Fine-Tuning requires 1,048,576 trainable parameters.

## 4. LoRA Calculation

LoRA trains only the two low-rank matrices A and B.

The number of trainable parameters is:

$$
N_{\text{LoRA}} = (r \times 1024) + (1024 \times r)
$$

Simplifying:

$$
\boxed{N_{\text{LoRA}} = 2048r}
$$

### 4.1. LoRA Rank = 4

$$
N_{\text{LoRA}} = 2048 \times 4 = 8,192
$$

Reduction ratio:

$$
\frac{1,048,576}{8,192} = 128
$$

Percentage reduction:

$$
\left(1-\frac{8,192}{1,048,576}\right)\times100
=99.21875\%
$$

### 4.2. LoRA Rank = 8

$$
N_{\text{LoRA}} = 2048 \times 8 = 16,384
$$

Reduction ratio:

$$
\frac{1,048,576}{16,384} = 64
$$

Percentage reduction:

$$
\left(1-\frac{16,384}{1,048,576}\right)\times100
=98.4375\%
$$

### 4.3. LoRA Rank = 16

$$
N_{\text{LoRA}} = 2048 \times 16 = 32,768
$$

Reduction ratio:

$$
\frac{1,048,576}{32,768} = 32
$$

Percentage reduction:

$$
\left(1-\frac{32,768}{1,048,576}\right)\times100
=96.875\%
$$

## 5. Comparison Summary

| Method | Trainable Parameters | Reduction Ratio | Parameter Reduction |
|---|---:|---:|---:|
| Full Fine-Tuning | 1,048,576 | 1× | 0% |
| LoRA (r = 4) | 8,192 | 128× | 99.21875% |
| LoRA (r = 8) | 16,384 | 64× | 98.4375% |
| LoRA (r = 16) | 32,768 | 32× | 96.875% |

The results show that LoRA significantly reduces the number of trainable parameters compared to Full Fine-Tuning.

A smaller rank results in fewer trainable parameters. However, a larger rank provides a higher-dimensional space for representing weight updates, potentially improving adaptation capacity.

## 6. Trainable Parameters vs. GPU Memory Consumption

Trainable parameter count and GPU memory consumption are different concepts.

Trainable parameters represent the number of model parameters updated during training, while GPU memory consumption includes all memory required for the training process.

| Memory Component | Full Fine-Tuning | LoRA |
|---|---|---|
| Original weights | Trainable | Frozen |
| Adapter weights | Not applicable | Trainable |
| Gradients | For original weights | For adapter weights |
| Optimizer states | For original weights | For adapter weights |
| Activations | Required | Required |
| Intermediate computations | Required | Required |

Even though LoRA freezes the original weight matrix, it must still keep the original weights available for forward computation.

For example, using FP32 precision, the original weight matrix requires:

$$
1024 \times 1024 \times 4 = 4,194,304 \text{ bytes}
$$

This is approximately 4 MiB.

LoRA must still store these original weights, even though they are not trainable.

Other GPU memory requirements include:

- Gradients for trainable adapter parameters.
- Optimizer states such as Adam momentum and variance.
- Forward and backward propagation activations.
- Temporary tensors and CUDA workspace.

Therefore, reducing trainable parameters does not mean GPU memory consumption decreases by exactly the same ratio.

LoRA reduces memory requirements associated with gradients and optimizer states, but the original model weights, activations, and other computational overhead remain.

## 7. Conclusion

For a 1024 × 1024 weight matrix, Full Fine-Tuning requires 1,048,576 trainable parameters.

LoRA requires significantly fewer parameters:

- Rank 4: 8,192 parameters (99.21875% reduction).
- Rank 8: 16,384 parameters (98.4375% reduction).
- Rank 16: 32,768 parameters (96.875% reduction).

LoRA provides a parameter-efficient alternative to Full Fine-Tuning by freezing the original weights and training low-rank adapter matrices.

However, trainable parameter reduction should not be confused with total GPU memory reduction, since frozen weights, activations, and other training overhead still consume GPU memory.