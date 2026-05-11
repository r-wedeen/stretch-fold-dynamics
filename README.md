# Stretch/Fold MLP Visualizer

Visualize a 1D absolute-value MLP as a material line that is repeatedly stretched and folded during training.

The project has two pieces:

- `scripts/train_export.py` trains a scalar stretch/fold network and exports snapshots.
- `viewer/index.html` plays stacked exported trajectories with movie controls and manual sliders.

## Run

Use the bundled Python runtime if your system Python does not have NumPy:

```bash
/Users/starpoint/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/train_export.py
```

Then open:

```text
viewer/index.html
```

The checked-in viewer loads `viewer/data/quartic.js` and `viewer/data/quintic.js`, so the HTML file can be opened directly without a dev server.

## Model

The default network is:

```text
h_0 = x
h_l = |a_l h_{l-1} + b_l|, l = 1..L
y = c h_L + d
```

The colors stay attached to the original input coordinate. As training progresses, the same colored material points move through the learned sequence of affine stretches and absolute-value folds.

The checked-in comparison shows `quartic`, a smooth double-well curve, followed by `quintic`, a scaled 5th-degree Chebyshev polynomial. Use `--target septic` for a sharper 7th-degree polynomial, `--target abs` for the simplest `f(x) = |x|` fold, `--target triangle` for a repeating triangle wave, or `--target sinmix` for a higher-frequency oscillatory mixture.

## Useful Options

```bash
python3 scripts/train_export.py --target triangle --layers 6 --steps 1500 --snapshots 1001
python3 scripts/train_export.py --target quartic --out viewer/data/quartic.js --global-name window.QUARTIC_TRAJECTORY
python3 scripts/train_export.py --target quintic --layers 6 --steps 1500 --snapshots 1001
python3 scripts/train_export.py --target quintic --out viewer/data/quintic.js --global-name window.QUINTIC_TRAJECTORY
python3 scripts/train_export.py --target septic --layers 6 --steps 1500 --snapshots 1001
python3 scripts/train_export.py --target sinmix --optimizer gd --lr 0.008 --steps 3000
python3 scripts/train_export.py --target abs --layers 20
python3 scripts/train_export.py --snapshot-schedule linear
python3 scripts/train_export.py --snapshot-power 3.0
python3 scripts/train_export.py --out viewer/data/trajectory.js
```

The default optimizer is Adam. Use `--optimizer gd` for calmer full-batch gradient descent, though it may fit less aggressively. The default snapshot schedule is linear with 1001 saved frames from step 0 to 1500. Use `--snapshot-schedule early` when you want a smaller export that concentrates frames near the start.

If PyTorch is installed, you can request it explicitly:

```bash
python3 scripts/train_export.py --backend torch
```

The `auto` backend uses PyTorch when available and otherwise falls back to the built-in NumPy trainer.
