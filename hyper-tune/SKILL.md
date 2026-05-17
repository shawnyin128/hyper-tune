---
name: hyper-tune
description: Lean hyperparameter tuning workflow for neural network training when compute is limited. Use when Codex is asked to tune model training; choose learning rate, optimizer, scheduler, warmup, weight decay, or architecture-specific initial training configs; plan LR/WD sweeps; review existing training configs; choose fine-tuning or LoRA/QLoRA starting hyperparameters; diagnose learning curves; debug stagnant, exploding, NaN, or abnormal initial loss; inspect overfitting or underfitting; or handle Chinese requests about 调参.
---

# Hyper Tune

## Overview

Use a cheap-to-expensive tuning sequence. First verify the loss and optimization path, then search learning rate and regularization with short runs, then train only the best candidates longer.

Keep the tuning loop evidence-driven: record the exact config, seed, data split, run length, train curve, validation curve, and stopping reason for every run. Never choose from test-set performance.

When the user needs a default starting config by model architecture or task type, read `references/initial-points.md` and adapt the table to the current codebase, batch size, and compute budget.

## Workflow

### 1. Establish the experiment frame

Identify the task, metric, model, optimizer, dataset split, batch size, compute budget, and current failure mode. If details are missing, state assumptions and proceed with the smallest useful probe.

Use this priority order:

1. Correctness checks before performance tuning.
2. Learning rate before broad architecture changes.
3. Weight decay and regularization after the model can optimize.
4. Longer runs only after short runs show useful signal.

### 2. Check initial loss

Turn off weight decay and other regularization. Evaluate the loss at initialization before trusting any training curve.

For a balanced `C`-class softmax classifier with random logits, expect an initial cross-entropy near `log(C)`. If the initial loss is far off, inspect labels, class count, loss reduction, logits shape, data normalization, masking, and target encoding before tuning hyperparameters.

Examples: binary classification should start near `log(2) ~= 0.69`, CIFAR-10-style 10-class classification near `log(10) ~= 2.30`, and ImageNet-style 1000-class classification near `log(1000) ~= 6.91`. For normal averaged cross-entropy classification, a deviation larger than roughly `0.1` is strong evidence to debug the implementation before training further.

Also confirm the first forward and backward pass produce finite activations, loss, and gradients.

### 3. Overfit a small sample

Train on a tiny sample, usually about 5-10 examples for image tasks or 5-10 minibatches when the dataloader setup makes that easier, and try to drive training accuracy close to 100% or training loss close to zero. Turn off regularization, heavy augmentation, dropout, mixup, label smoothing, and early stopping for this probe.

Treat this as an optimization sanity check only. Do not count small-sample accuracy or loss as full-data validation performance, and do not use it to select final hyperparameters.

Interpret the result:

- Loss does not go down: learning rate may be too low, gradients may not flow, initialization may be poor, or the training loop may be wrong.
- Loss becomes `Inf` or `NaN`: learning rate may be too high, initialization may be poor, normalization may be unstable, or gradients may explode.
- Small sample cannot be overfit after reasonable LR changes: suspect data labels, architecture capacity, objective mismatch, frozen parameters, or optimizer wiring.

Do not move to full-data sweeps until this probe passes or the failure is explicitly understood.

### 4. Find a learning rate that decreases loss quickly

Use the architecture and training loop from the overfit probe. Switch to the full training set, enable only small weight decay if needed, and run short trials for roughly 100 iterations.

Try logarithmic learning rates such as `1e-1`, `1e-2`, `1e-3`, and `1e-4`, adjusted for the optimizer and model family. Keep the run short enough to test multiple values cheaply.

Typical starting ranges: for SGD with momentum, try around `1e-1` to `1e-2`; for Adam or AdamW, try around `1e-3` to `1e-4`.

Choose the largest learning rate that makes loss drop clearly without instability. If all curves are flat, move LR up or revisit the small-sample check. If curves explode, move LR down by 3-10x and inspect numerical stability.

Track the update-to-weight scale as a sanity check. For SGD-like updates:

```python
param_scale = np.linalg.norm(W.ravel())
update = -learning_rate * dW
update_scale = np.linalg.norm(update.ravel())
ratio = update_scale / (param_scale + 1e-12)
```

Treat ratios around `1e-3` to `1e-2` as plausible starting evidence, not a universal rule. For adaptive optimizers, prefer measuring the actual parameter delta before and after `optimizer.step()` instead of approximating with raw gradients.

### 5. Run a coarse grid

Choose 3-4 learning rates around the best short-run LR and combine them with a small weight-decay grid. Good starter weight decay values are `1e-4`, `5e-4`, and `1e-3`; include `0` or `1e-5` when checking whether regularization is hurting optimization, unless the model family has a known better range.

Keep scheduler behavior simple during this phase. Prefer fixed LR for the coarse grid unless the model family requires warmup for stability; add cosine, step decay, or other schedules only after the raw fixed-LR behavior is understood.

Train each run for about 1-5 epochs. Compare validation metrics, train metrics, learning curves, and stability. Stop clearly bad runs early, but keep enough evidence to explain why they failed.

Prefer a compact table:

| Run | LR | Weight decay | Epochs/iters | Train metric | Val metric | Curve diagnosis | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |

### 6. Refine the grid and train longer

Pick the best candidates from the coarse grid. Train them longer, often around 10-20 epochs first, initially without learning-rate decay so the raw optimization behavior remains visible.

After the fixed-LR behavior is understood, run the final full-budget jobs for the task's normal length, often tens to hundreds of epochs for vision classification, and add a schedule such as cosine decay only when the curve supports it. If loss plateaus, learning-rate decay can help. If loss is still decreasing well, delaying decay is usually better.

### 7. Read learning curves before changing knobs

Plot noisy training loss as both scatter points and a moving average. Plot train and validation metrics on the same axes when possible.

Use these diagnoses:

- Loss is flat for a long time, then suddenly drops: suspect bad initialization, bad warmup, optimizer state, or an LR that only becomes effective late.
- Loss plateaus: try learning-rate decay or a modest LR reduction.
- Loss was still going down when LR dropped: decay happened too early.
- Accuracy is still rising: train longer before changing many knobs.
- Huge train/validation gap: overfitting; increase regularization, add data or augmentation, reduce capacity, or use earlier stopping.
- Small train/validation gap with low performance: underfitting; train longer, use a bigger model, reduce excessive regularization, or try a higher LR.
- Validation degrades while train improves: overfitting or distribution mismatch; inspect split quality and regularization.

### 8. Finalize the run

Use the selected config for a full-budget training run. Save the exact configuration, code version, seed, checkpoint policy, and final curves.

When budget allows, rerun the chosen configuration with multiple seeds and report mean and variance. Do not treat a single lucky validation result as a stable conclusion.

## Response Pattern

When using this skill, return:

1. Current diagnosis from available evidence.
2. The next cheapest probe to run.
3. A compact sweep plan with LR, weight decay, run length, and stop criteria.
4. What result would confirm or refute each hypothesis.
5. What not to tune yet, if the sanity checks are not passed.

Keep recommendations ordered. Avoid broad unconstrained sweeps unless the user explicitly has enough compute.

The first three steps are intentionally cheap. Run them before long training because they often eliminate most bad configurations and implementation errors.
