#!/usr/bin/env python3
"""Train a 1D absolute-value MLP and export viewer-ready snapshots."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


def target_values_np(np, x, name: str):
    if name == "abs":
        return np.abs(x)
    if name == "sinmix":
        return np.sin(2.2 * x) + 0.22 * np.cos(5.1 * x)
    if name == "triangle":
        period = 1.6
        u = (x / period) - np.floor((x / period) + 0.5)
        return 1.0 - 4.0 * np.abs(u)
    if name == "bump":
        return 1.2 * np.exp(-1.4 * (x + 0.8) ** 2) - 0.9 * np.exp(-2.2 * (x - 0.9) ** 2)
    if name == "quadratic":
        return 0.45 * x * x - 0.8
    raise ValueError(f"unknown target {name!r}")


def target_values_torch(torch, x, name: str):
    if name == "abs":
        return torch.abs(x)
    if name == "sinmix":
        return torch.sin(2.2 * x) + 0.22 * torch.cos(5.1 * x)
    if name == "triangle":
        period = 1.6
        u = (x / period) - torch.floor((x / period) + 0.5)
        return 1.0 - 4.0 * torch.abs(u)
    if name == "bump":
        return 1.2 * torch.exp(-1.4 * (x + 0.8) ** 2) - 0.9 * torch.exp(-2.2 * (x - 0.9) ** 2)
    if name == "quadratic":
        return 0.45 * x * x - 0.8
    raise ValueError(f"unknown target {name!r}")


@dataclass
class Snapshot:
    step: int
    loss: float
    params: Dict[str, List[float]]
    layers: List[List[float]]
    pred: List[float]


def rounded(values, digits: int = 5):
    return [round(float(v), digits) for v in values]


def init_numpy_params(np, layers: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.normal(1.15, 0.32, size=layers)
    signs = np.where(rng.random(layers) < 0.5, -1.0, 1.0)
    a = a * signs
    b = rng.normal(0.0, 0.35, size=layers)
    c = np.array(rng.normal(0.8, 0.25))
    d = np.array(rng.normal(0.0, 0.15))
    return {"a": a, "b": b, "c": c, "d": d}


def snapshot_step_set(np, args):
    if args.snapshot_schedule == "linear":
        steps = np.linspace(0, args.steps, args.snapshots)
    elif args.snapshot_schedule == "early":
        t = np.linspace(0.0, 1.0, args.snapshots)
        steps = args.steps * (t**args.snapshot_power)
    else:
        raise ValueError(f"unknown snapshot schedule {args.snapshot_schedule!r}")
    return set(np.unique(np.rint(steps).astype(int)).tolist())


def forward_numpy(np, x, params):
    hs = [x]
    zs = []
    h = x
    for a, b in zip(params["a"], params["b"]):
        z = a * h + b
        h = np.abs(z)
        zs.append(z)
        hs.append(h)
    y = params["c"] * h + params["d"]
    return hs, zs, y


def train_numpy(args) -> Tuple[List[float], List[float], List[Snapshot], Dict[str, object]]:
    import numpy as np

    x = np.linspace(args.xmin, args.xmax, args.samples)
    target = target_values_np(np, x, args.target)
    params = init_numpy_params(np, args.layers, args.seed)
    snapshot_steps = snapshot_step_set(np, args)
    snapshots: List[Snapshot] = []

    moments = {name: np.zeros_like(value, dtype=float) for name, value in params.items()}
    velocities = {name: np.zeros_like(value, dtype=float) for name, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def capture(step: int, loss: float):
        hs, _, pred = forward_numpy(np, x, params)
        snapshots.append(
            Snapshot(
                step=step,
                loss=loss,
                params={
                    "a": rounded(params["a"]),
                    "b": rounded(params["b"]),
                    "c": [round(float(params["c"]), 5)],
                    "d": [round(float(params["d"]), 5)],
                },
                layers=[rounded(h) for h in hs],
                pred=rounded(pred),
            )
        )

    for step in range(args.steps + 1):
        hs, zs, pred = forward_numpy(np, x, params)
        err = pred - target
        loss = float(np.mean(err * err))
        if step in snapshot_steps:
            capture(step, loss)
        if step == args.steps:
            break

        n = float(x.shape[0])
        grad_y = 2.0 * err / n
        grads = {
            "a": np.zeros_like(params["a"]),
            "b": np.zeros_like(params["b"]),
            "c": np.array(np.sum(grad_y * hs[-1])),
            "d": np.array(np.sum(grad_y)),
        }
        grad_h = grad_y * params["c"]

        for idx in reversed(range(args.layers)):
            grad_z = grad_h * np.sign(zs[idx])
            grads["a"][idx] = np.sum(grad_z * hs[idx])
            grads["b"][idx] = np.sum(grad_z)
            grad_h = grad_z * params["a"][idx]

        t = step + 1
        for name in params:
            moments[name] = beta1 * moments[name] + (1.0 - beta1) * grads[name]
            velocities[name] = beta2 * velocities[name] + (1.0 - beta2) * (grads[name] ** 2)
            m_hat = moments[name] / (1.0 - beta1**t)
            v_hat = velocities[name] / (1.0 - beta2**t)
            params[name] = params[name] - args.lr * m_hat / (np.sqrt(v_hat) + eps)

    meta = {"backend": "numpy", "target": args.target}
    return rounded(x), rounded(target), snapshots, meta


def train_torch(args) -> Tuple[List[float], List[float], List[Snapshot], Dict[str, object]]:
    import torch

    torch.manual_seed(args.seed)
    x = torch.linspace(args.xmin, args.xmax, args.samples)
    target = target_values_torch(torch, x, args.target)

    a = torch.nn.Parameter(torch.randn(args.layers) * 0.32 + 1.15)
    signs = torch.where(torch.rand(args.layers) < 0.5, -1.0, 1.0)
    with torch.no_grad():
        a.mul_(signs)
    b = torch.nn.Parameter(torch.randn(args.layers) * 0.35)
    c = torch.nn.Parameter(torch.randn(()) * 0.25 + 0.8)
    d = torch.nn.Parameter(torch.randn(()) * 0.15)
    opt = torch.optim.Adam([a, b, c, d], lr=args.lr)
    import numpy as np

    snapshot_steps = snapshot_step_set(np, args)
    snapshots: List[Snapshot] = []

    def forward():
        hs = [x]
        h = x
        for idx in range(args.layers):
            h = torch.abs(a[idx] * h + b[idx])
            hs.append(h)
        return hs, c * h + d

    def capture(step: int, loss: float):
        with torch.no_grad():
            hs, pred = forward()
            snapshots.append(
                Snapshot(
                    step=step,
                    loss=loss,
                    params={
                        "a": rounded(a.detach().cpu().tolist()),
                        "b": rounded(b.detach().cpu().tolist()),
                        "c": [round(float(c.detach().cpu()), 5)],
                        "d": [round(float(d.detach().cpu()), 5)],
                    },
                    layers=[rounded(h.detach().cpu().tolist()) for h in hs],
                    pred=rounded(pred.detach().cpu().tolist()),
                )
            )

    for step in range(args.steps + 1):
        opt.zero_grad()
        _, pred = forward()
        loss_tensor = torch.mean((pred - target) ** 2)
        loss = float(loss_tensor.detach().cpu())
        if step in snapshot_steps:
            capture(step, loss)
        if step == args.steps:
            break
        loss_tensor.backward()
        opt.step()

    meta = {"backend": "torch", "target": args.target, "torchVersion": torch.__version__}
    return rounded(x.detach().cpu().tolist()), rounded(target.detach().cpu().tolist()), snapshots, meta


def build_payload(args, x, target, snapshots, meta):
    losses = [{"step": snap.step, "loss": round(float(snap.loss), 8)} for snap in snapshots]
    return {
        "meta": {
            **meta,
            "layers": args.layers,
            "samples": args.samples,
            "steps": args.steps,
            "snapshots": len(snapshots),
            "snapshotSchedule": args.snapshot_schedule,
            "snapshotPower": args.snapshot_power,
            "xmin": args.xmin,
            "xmax": args.xmax,
            "lr": args.lr,
            "seed": args.seed,
        },
        "x": x,
        "target": target,
        "losses": losses,
        "snapshots": [
            {
                "step": snap.step,
                "loss": round(float(snap.loss), 8),
                "params": snap.params,
                "layers": snap.layers,
                "pred": snap.pred,
            }
            for snap in snapshots
        ],
    }


def write_js(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("window.TRAJECTORY = ")
        json.dump(payload, handle, separators=(",", ":"))
        handle.write(";\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--target", choices=["abs", "sinmix", "triangle", "bump", "quadratic"], default="sinmix")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--samples", type=int, default=520)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--snapshots", type=int, default=251)
    parser.add_argument("--snapshot-schedule", choices=["early", "linear"], default="linear")
    parser.add_argument("--snapshot-power", type=float, default=2.4)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--xmin", type=float, default=-2.8)
    parser.add_argument("--xmax", type=float, default=2.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="viewer/data/trajectory.js")
    return parser.parse_args()


def main():
    args = parse_args()
    backend = args.backend
    if backend == "auto":
        try:
            import torch  # noqa: F401

            backend = "torch"
        except Exception:
            backend = "numpy"

    if backend == "torch":
        x, target, snapshots, meta = train_torch(args)
    else:
        x, target, snapshots, meta = train_numpy(args)

    payload = build_payload(args, x, target, snapshots, meta)
    write_js(args.out, payload)
    first = snapshots[0].loss
    last = snapshots[-1].loss
    print(f"wrote {args.out}")
    print(f"backend={payload['meta']['backend']} snapshots={len(snapshots)} loss={first:.6f}->{last:.6f}")


if __name__ == "__main__":
    main()
