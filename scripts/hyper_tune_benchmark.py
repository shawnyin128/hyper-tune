import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrialResult:
    model: str
    strategy: str
    phase: str
    trial_index: int
    optimizer: str
    lr: float
    weight_decay: float
    scheduler: str
    epochs: int
    steps: int
    seconds: float
    initial_loss: float
    final_train_loss: float
    final_val_acc: float
    best_val_acc: float
    hit_target: bool
    notes: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_mlp_data(seed: int, n_train: int = 3072, n_val: int = 1024):
    gen = torch.Generator().manual_seed(seed)
    dim = 32
    classes = 3
    centers = torch.randn(classes, dim, generator=gen) * 2.5

    def sample(n):
        y = torch.randint(0, classes, (n,), generator=gen)
        x = centers[y] + 1.15 * torch.randn(n, dim, generator=gen)
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        return x.float(), y.long()

    return (*sample(n_train), *sample(n_val), classes)


def make_cnn_data(seed: int, n_train: int = 4096, n_val: int = 1024):
    gen = torch.Generator().manual_seed(seed)
    classes = 4
    h = w = 16

    def sample(n):
        y = torch.randint(0, classes, (n,), generator=gen)
        x = 0.35 * torch.randn(n, 1, h, w, generator=gen)
        for i, label in enumerate(y.tolist()):
            if label == 0:
                x[i, 0, :, 3:6] += 1.2
            elif label == 1:
                x[i, 0, 10:13, :] += 1.2
            elif label == 2:
                diag = torch.arange(h)
                x[i, 0, diag, diag] += 1.2
                x[i, 0, diag, torch.clamp(diag + 1, max=w - 1)] += 0.7
            else:
                x[i, 0, 4:12, 4:12] += 0.65
                x[i, 0, 6:10, 6:10] -= 0.7
        x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        return x.float(), y.long()

    return (*sample(n_train), *sample(n_val), classes)


def make_transformer_data(seed: int, n_train: int = 4096, n_val: int = 1024):
    gen = torch.Generator().manual_seed(seed)
    seq_len = 18
    vocab = 32
    classes = 2

    def sample(n):
        x = torch.randint(0, vocab, (n, seq_len), generator=gen)
        first = x[:, : seq_len // 2].float().mean(1)
        second = x[:, seq_len // 2 :].float().mean(1)
        y = (first > second + 0.6).long()
        # Inject a sparse motif so attention has a useful shortcut.
        motif = torch.rand(n, generator=gen) < 0.35
        x[motif, 2] = 29
        x[motif, 13] = 3
        y[motif] = 1
        return x.long(), y.long()

    x_train, y_train = sample(n_train)
    x_val, y_val = sample(n_val)
    return x_train, y_train, x_val, y_val, classes


class MLP(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class TinyTransformer(nn.Module):
    def __init__(self, out_dim, vocab=32, seq_len=18):
        super().__init__()
        width = 64
        self.token = nn.Embedding(vocab, width)
        self.pos = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(width, out_dim)

    def forward(self, x):
        h = self.token(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h.mean(1))


def build_model(name, classes):
    if name == "mlp":
        return MLP(classes)
    if name == "cnn":
        return TinyCNN(classes)
    if name == "transformer":
        return TinyTransformer(classes)
    raise ValueError(name)


def make_data(name, seed):
    if name == "mlp":
        return make_mlp_data(seed)
    if name == "cnn":
        return make_cnn_data(seed)
    if name == "transformer":
        return make_transformer_data(seed)
    raise ValueError(name)


def make_optimizer(params, name, lr, wd):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


def make_scheduler(opt, name, epochs):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[max(1, epochs // 2)], gamma=0.1)
    raise ValueError(name)


def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def initial_loss(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        x, y = next(iter(loader))
        return loss_fn(model(x.to(device)), y.to(device)).item()


def train_once(
    model_name,
    data,
    config,
    target,
    device,
    seed,
    max_epochs,
    batch_size,
    phase,
    strategy,
    trial_index,
    max_steps=None,
):
    set_seed(seed)
    x_train, y_train, x_val, y_val, classes = data
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    model = build_model(model_name, classes).to(device)
    init_loss = initial_loss(model, train_loader, device)
    opt = make_optimizer(model.parameters(), config["optimizer"], config["lr"], config["weight_decay"])
    sched = make_scheduler(opt, config.get("scheduler", "none"), max_epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    steps = 0
    last_loss = init_loss
    start = time.perf_counter()
    hit = False

    for _epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                seconds = time.perf_counter() - start
                return TrialResult(
                    model_name,
                    strategy,
                    phase,
                    trial_index,
                    config["optimizer"],
                    config["lr"],
                    config["weight_decay"],
                    config.get("scheduler", "none"),
                    _epoch + 1,
                    steps,
                    seconds,
                    init_loss,
                    float("nan"),
                    0.0,
                    best,
                    False,
                    "nonfinite_loss",
                )
            loss.backward()
            if config.get("clip", 0):
                nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
            opt.step()
            steps += 1
            last_loss = loss.item()
            if max_steps is not None and steps >= max_steps:
                break
        if sched is not None:
            sched.step()
        val = eval_acc(model, val_loader, device)
        best = max(best, val)
        if val >= target:
            hit = True
            break
        if max_steps is not None and steps >= max_steps:
            break

    seconds = time.perf_counter() - start
    final_val = eval_acc(model, val_loader, device)
    best = max(best, final_val)
    return TrialResult(
        model_name,
        strategy,
        phase,
        trial_index,
        config["optimizer"],
        config["lr"],
        config["weight_decay"],
        config.get("scheduler", "none"),
        max_epochs,
        steps,
        seconds,
        init_loss,
        last_loss,
        final_val,
        best,
        hit or best >= target,
        "",
    )


def tiny_overfit(model_name, data, base_config, device, seed, batch_size):
    x_train, y_train, x_val, y_val, classes = data
    n = min(16, x_train.shape[0])
    tiny_data = (x_train[:n], y_train[:n], x_train[:n], y_train[:n], classes)
    return train_once(
        model_name,
        tiny_data,
        base_config,
        target=0.98,
        device=device,
        seed=seed,
        max_epochs=20,
        batch_size=min(batch_size, n),
        phase="tiny_overfit",
        strategy="skill",
        trial_index=0,
    )


def lr_probe(model_name, data, optimizer, lrs, device, seed, batch_size):
    probe_results = []
    for i, lr in enumerate(lrs, 1):
        cfg = {"optimizer": optimizer, "lr": lr, "weight_decay": 0.0, "scheduler": "none", "clip": 1.0}
        res = train_once(
            model_name,
            data,
            cfg,
            target=0.999,
            device=device,
            seed=seed + i,
            max_epochs=3,
            batch_size=batch_size,
            phase="lr_probe",
            strategy="skill",
            trial_index=i,
            max_steps=100,
        )
        probe_results.append(res)
        if math.isfinite(res.final_train_loss) and res.final_train_loss < 0.35 * max(res.initial_loss, 1e-9):
            break
    finite = [r for r in probe_results if math.isfinite(r.final_train_loss)]
    if not finite:
        return probe_results, lrs[-1]
    # Prefer the highest LR that makes clear progress without instability.
    scored = sorted(
        finite,
        key=lambda r: (r.final_train_loss / max(r.initial_loss, 1e-9), -r.lr),
    )
    return probe_results, scored[0].lr


def naive_space(model_name):
    if model_name == "cnn":
        opts = ["adamw", "sgd"]
        lrs = [1e-4, 1e-3, 1e-2, 1e-1]
        wds = [0.0, 1e-4, 5e-4, 1e-2]
    elif model_name == "transformer":
        opts = ["sgd", "adamw"]
        lrs = [1e-4, 1e-3, 1e-2, 1e-1]
        wds = [0.0, 1e-4, 1e-2, 1e-1]
    else:
        opts = ["sgd", "adamw", "adam"]
        lrs = [1e-4, 1e-3, 1e-2, 1e-1]
        wds = [0.0, 1e-4, 1e-3, 1e-2]
    configs = []
    for opt in opts:
        for lr in lrs:
            for wd in wds:
                configs.append({"optimizer": opt, "lr": lr, "weight_decay": wd, "scheduler": "none", "clip": 1.0})
    return configs


def skill_defaults(model_name):
    if model_name == "cnn":
        return "sgd", [1e-1, 1e-2, 1e-3, 1e-4], [1e-4, 5e-4, 1e-3], "none"
    if model_name == "transformer":
        return "adamw", [1e-3, 3e-3, 1e-2, 3e-4], [0.0, 1e-2, 1e-1], "none"
    return "adamw", [1e-3, 3e-4, 1e-4, 3e-3], [0.0, 1e-4, 1e-3], "none"


def run_strategy(model_name, strategy, data, target, device, seed, batch_size):
    results = []
    if strategy == "naive":
        configs = naive_space(model_name)
        random.Random(seed + 12345).shuffle(configs)
        for i, cfg in enumerate(configs, 1):
            res = train_once(
                model_name,
                data,
                cfg,
                target=target,
                device=device,
                seed=seed + i,
                max_epochs=5,
                batch_size=batch_size,
                phase="broad_grid",
                strategy="naive",
                trial_index=i,
            )
            results.append(res)
            if res.best_val_acc >= target:
                break
        return results

    optimizer, probe_lrs, wds, sched = skill_defaults(model_name)
    probes, best_lr = lr_probe(model_name, data, optimizer, probe_lrs, device, seed + 2000, batch_size)
    results.extend(probes)
    candidate_lrs = sorted(set([best_lr / 3, best_lr, best_lr * 3]))
    trial = 1
    coarse_results = []
    for lr in candidate_lrs:
        for wd in wds:
            cfg = {"optimizer": optimizer, "lr": lr, "weight_decay": wd, "scheduler": sched, "clip": 1.0}
            res = train_once(
                model_name,
                data,
                cfg,
                target=target,
                device=device,
                seed=seed + 3000 + trial,
                max_epochs=5,
                batch_size=batch_size,
                phase="coarse_grid",
                strategy="skill",
                trial_index=trial,
            )
            results.append(res)
            coarse_results.append((res.best_val_acc, cfg))
            trial += 1
            if res.best_val_acc >= target:
                return results
    for _, cfg in sorted(coarse_results, key=lambda item: item[0], reverse=True)[:2]:
        refine_cfg = dict(cfg)
        refine_cfg["scheduler"] = "cosine"
        res = train_once(
            model_name,
            data,
            refine_cfg,
            target=target,
            device=device,
            seed=seed + 4000 + trial,
            max_epochs=10,
            batch_size=batch_size,
            phase="refine_grid",
            strategy="skill",
            trial_index=trial,
        )
        results.append(res)
        trial += 1
        if res.best_val_acc >= target:
            return results
    return results


def summarize(results, target):
    total_steps = sum(r.steps for r in results)
    total_seconds = sum(r.seconds for r in results)
    full_results = [r for r in results if r.phase in {"broad_grid", "coarse_grid", "refine_grid"}]
    best = max((r.best_val_acc for r in full_results), default=0.0)
    hit_at = next((i for i, r in enumerate(full_results, 1) if r.best_val_acc >= target), None)
    train_trials = len(full_results)
    return {
        "total_steps": total_steps,
        "total_seconds": total_seconds,
        "best_val_acc": best,
        "hit_target": hit_at is not None,
        "hit_at_trial": hit_at or len(full_results),
        "train_trials": train_trials,
        "all_trials": len(results),
    }


def plot_summary(summary_rows, out_dir):
    labels = sorted({r["model"] for r in summary_rows})
    metrics = [
        ("total_steps", "Training steps to target / stop"),
        ("total_seconds", "Wall time to target / stop (s)"),
        ("all_trials", "Trials including probes"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    width = 0.36
    x = np.arange(len(labels))
    for ax, (metric, title) in zip(axes, metrics):
        naive = [next(r for r in summary_rows if r["model"] == m and r["strategy"] == "naive")[metric] for m in labels]
        skill = [next(r for r in summary_rows if r["model"] == m and r["strategy"] == "skill")[metric] for m in labels]
        ax.bar(x - width / 2, naive, width, label="naive")
        ax.bar(x + width / 2, skill, width, label="hyper-tune")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=180)

    fig, ax = plt.subplots(figsize=(8, 4))
    for model in labels:
        naive = next(r for r in summary_rows if r["model"] == model and r["strategy"] == "naive")
        skill = next(r for r in summary_rows if r["model"] == model and r["strategy"] == "skill")
        speedup = naive["total_steps"] / max(1, skill["total_steps"])
        ax.bar(model, speedup)
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_ylabel("Step reduction factor (naive / hyper-tune)")
    ax.set_title("Search-cost reduction from hyper-tune-guided search")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup.png", dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    # Targets are intentionally high enough that a lucky first broad-grid trial
    # does not count as "good enough"; the benchmark is about finding strong
    # hyperparameters, not merely clearing an easy sanity threshold.
    targets = {"mlp": 0.97, "cnn": 0.99, "transformer": 0.89}
    batch_sizes = {"mlp": 128, "cnn": 128, "transformer": 128}
    all_results = []
    summary_rows = []

    for model_name in ["mlp", "cnn", "transformer"]:
        data = make_data(model_name, args.seed + 17)
        for strategy in ["naive", "skill"]:
            results = run_strategy(
                model_name,
                strategy,
                data,
                targets[model_name],
                device,
                args.seed + (0 if strategy == "naive" else 50000),
                batch_sizes[model_name],
            )
            all_results.extend(results)
            s = summarize(results, targets[model_name])
            s.update({"model": model_name, "strategy": strategy, "target": targets[model_name]})
            summary_rows.append(s)
            print(json.dumps(s, sort_keys=True))

    with (args.out / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrialResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.__dict__)

    with (args.out / "summary.csv").open("w", newline="") as f:
        fields = ["model", "strategy", "target", "hit_target", "best_val_acc", "total_steps", "total_seconds", "hit_at_trial", "train_trials", "all_trials"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r[k] for k in fields})

    with (args.out / "summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)

    plot_summary(summary_rows, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
