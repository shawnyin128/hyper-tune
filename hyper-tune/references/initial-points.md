# Initial Hyperparameter Points

Use this table to choose the first reasonable training config before running the cheap sanity checks in `SKILL.md`. Treat every value as an initial point, not a guarantee. Scale learning rate with effective batch size only when the optimizer and model family support that convention, and validate with the first 100-step LR probe.

## Architecture And Task Table

| Model / task family | Optimizer default | Starting LR | Weight decay | Scheduler / warmup | Typical run shape | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| CNN image classification, e.g. ResNet on CIFAR/ImageNet | SGD + momentum `0.9`, often Nesterov off by default | `0.1` for batch 256, or `0.01` for smaller batches; scale roughly linearly with batch | `1e-4` or `5e-4` | Step decay or cosine decay; warmup 0-5 epochs for larger batch/ImageNet | 90-300 epochs | Strong baseline for classic convnets. If loss explodes, try `0.03` or `0.01`. |
| Small CNNs on small datasets | SGD + momentum or AdamW | SGD `1e-2` to `1e-1`; AdamW `1e-3` | `1e-4`, `5e-4`, or `0` | Cosine or step decay; short/no warmup | 50-200 epochs | Regularization and augmentation can dominate; first overfit a tiny subset. |
| ConvNeXt / modern convnets | AdamW or SGD + momentum depending on recipe | AdamW `4e-4` to `1e-3`; SGD `0.05` to `0.1` | AdamW `0.01` to `0.05`; SGD `1e-4` | Cosine decay with 5-20 epoch warmup | 100-300 epochs | AdamW recipes often use stronger augmentation and higher weight decay than ResNet SGD recipes. |
| Vision Transformer / DeiT / Swin classification | AdamW, betas around `(0.9, 0.999)` | `5e-4` to `1e-3` for moderate batches; `1e-4` to `3e-4` for fine-tuning | `0.03` to `0.1`, often `0.05` | Cosine decay with 5-20 epoch warmup | 100-300 epochs from scratch; 10-100 epochs fine-tune | Needs warmup, augmentation, and regularization. Exclude bias and norm params from weight decay when supported. |
| Object detection, Faster R-CNN / Mask R-CNN style | SGD + momentum `0.9` | `0.02` for total batch 16 images; scale down for smaller batch | `1e-4` | Step decay or multi-step schedule; short warmup hundreds to ~1000 iters | 1x/2x/3x detection schedules | Detection LR is often specified by total images per batch, not per device. |
| Object detection, DETR-style transformer detector | AdamW | Backbone `1e-5`, transformer/head `1e-4` | `1e-4` | Step or cosine; warmup optional/short | 50-300 epochs depending on variant | Use parameter groups: lower LR for pretrained backbone. |
| Segmentation U-Net / CNN encoder-decoder | AdamW or Adam | `1e-3` for Adam/AdamW; `1e-4` for fragile medical/small data | `1e-5` to `1e-4` | Cosine, polynomial decay, or ReduceLROnPlateau | 50-300 epochs | Dice/BCE/CE losses can change loss scale; watch validation metric, not only loss. |
| Transformer encoder fine-tuning, e.g. BERT/RoBERTa classification | AdamW, betas `(0.9, 0.999)`, eps `1e-8` | `1e-5` to `5e-5`, common start `2e-5` or `3e-5` | `0.01` | Linear decay with 6-10% warmup, or cosine with warmup | 2-10 epochs | Use lower LR for small data or unstable tasks. Exclude bias and LayerNorm from weight decay. |
| Small transformer from scratch, e.g. toy sequence classifier | AdamW | `1e-3` to `1e-2`, start around `1e-3` then probe upward | `0` to `0.1` | Keep fixed LR during coarse search; add cosine only for longer refine runs | 5-50 epochs | Do not blindly use BERT fine-tuning LR. For shallow randomly initialized transformers, `1e-5` to `5e-5` is often far too small. |
| Transformer encoder pretraining / MLM | AdamW | `1e-4` to `5e-4` depending on batch/model size | `0.01` to `0.1` | Linear or cosine decay with 1-10% warmup | Step-budget driven | Larger effective batch usually supports larger LR; use gradient clipping around `1.0`. |
| Decoder-only LLM pretraining | AdamW or fused AdamW, betas often `(0.9, 0.95)` | `1e-4` to `3e-4` for small/medium models; lower for very large models | `0.1` typical | Cosine or linear decay; warmup 1-3% of steps | Token-budget driven | Clip grad norm around `1.0`. Exclude norm/bias from weight decay when recipe does so. |
| Decoder-only LLM full fine-tuning | AdamW | `5e-6` to `2e-5` | `0` to `0.1`, often `0.01` | Cosine or linear decay with 3-10% warmup | 1-5 epochs | Use conservative LR; monitor overfitting and catastrophic forgetting. |
| LLM LoRA / QLoRA instruction tuning | AdamW variants, paged AdamW for QLoRA | `1e-4` to `3e-4`; conservative start `2e-4` | `0` to `0.01`, often `0` | Cosine or linear decay with 3-10% warmup | 1-5 epochs | Tune LoRA rank/alpha/dropout after LR works. Higher LR than full fine-tune is common. |
| Diffusion U-Net training | AdamW, betas often `(0.9, 0.999)` | `1e-4` common; `5e-5` for unstable large runs | `0` to `1e-2`, often low | Constant, cosine, or linear decay; warmup 500-10k steps | Step-budget driven | EMA of weights is common. Loss scale may not correlate perfectly with sample quality. |
| GAN training | Adam or AdamW, betas often `(0.5, 0.999)` or `(0.0, 0.99)` by recipe | `2e-4` common; sometimes `1e-4` | `0` or very low | Usually constant or mild decay | Step-budget driven | Balance generator/discriminator updates before broad sweeps. Curves are less diagnostic than samples/metrics. |
| Tabular MLP / tabular transformer | AdamW | `1e-3` to `3e-4` | `1e-5` to `1e-2` | Cosine, ReduceLROnPlateau, or constant | 50-300 epochs with early stopping | Strong regularization can hurt small data. Compare against non-neural baselines when possible. |
| Graph neural networks | Adam or AdamW | `1e-2` to `1e-3` | `5e-4`, `1e-4`, or `0` | Constant, step decay, or ReduceLROnPlateau | 100-1000 epochs depending on graph/task | Weight decay around `5e-4` is a common citation-era GCN baseline; tune dropout too. |
| Time-series CNN/RNN/Transformer | AdamW or Adam | `1e-3` to `1e-4` | `1e-5` to `1e-2` | Cosine, one-cycle, or ReduceLROnPlateau | 20-200 epochs | Validate splits carefully to avoid leakage; tune windowing and normalization before over-sweeping LR. |
| Reinforcement learning policy/value nets | Adam | `3e-4` common for PPO-style; `1e-4` to `1e-3` by algorithm | Usually `0` or low | Often constant or linear anneal | Environment-step driven | Algorithm hyperparameters can dominate optimizer choices; tune entropy, clip range, target update, and rollout length with the optimizer. |

## Cross-Cutting Defaults

| Component | Default initial point | When to change |
| --- | --- | --- |
| AdamW betas | `(0.9, 0.999)` for most fine-tuning; `(0.9, 0.95)` is common for LLM pretraining | Lower beta2 can help large-scale transformer training respond faster. |
| AdamW epsilon | `1e-8` | Increase only for numerical stability issues in mixed precision. |
| SGD momentum | `0.9` | Try `0.95` only after LR is stable; lower if oscillation persists. |
| Gradient clipping | `1.0` global norm for transformers/RNNs/LLMs | Add when gradients spike, mixed precision is unstable, or loss occasionally explodes. |
| Warmup | 0 for small SGD CNN runs; 3-10% of steps for transformer fine-tuning; 1-3% for LLM pretraining | Increase if early loss spikes or the run is unstable; reduce if learning starts too late. |
| Weight decay exclusions | Exclude bias and norm parameters for AdamW transformer-style recipes | Keep simple if the codebase lacks clean parameter groups, but add exclusions when over-regularizing norms/biases is plausible. |
| Batch LR scaling | Linear scaling is a starting heuristic for SGD CNNs; use more caution for AdamW | Always re-run the short LR probe after changing effective batch size. |
| Mixed precision | Use bf16 when hardware supports it; fp16 may need loss scaling | If NaNs appear, check precision, gradient scale, normalization layers, and LR. |
| EMA | Common in diffusion, some vision recipes | Add after baseline trains correctly; it changes evaluation behavior. |

## First Sweep Templates

For an unknown neural training run, start narrow:

| Probe | Values |
| --- | --- |
| LR range | Four log points around the architecture default, e.g. `[base/10, base/3, base, base*3]` |
| Weight decay | Three values around the family default, plus `0` when regularization may block optimization |
| Warmup | Try default warmup and half/default/double only if early curves show instability or delayed learning |
| Scheduler | Hold scheduler fixed during coarse LR/WD sweeps unless the scheduler is the suspected issue |
| Seeds | Use one seed for cheap filtering; use 3+ seeds only for finalists |

When the table conflicts with the repository's established recipe, prefer the repository recipe as the first candidate and use this table to choose nearby alternatives.
