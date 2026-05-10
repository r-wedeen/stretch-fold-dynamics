# Stretch/Fold MLP Visualizer

Visualize a 1D absolute-value MLP as a material line that is repeatedly stretched and folded during training.

The project has two pieces:

- `scripts/train_export.py` trains a scalar stretch/fold network and exports snapshots.
- `viewer/index.html` plays the exported trajectory with a movie control and manual slider.

## Run

Use the bundled Python runtime if your system Python does not have NumPy:

```bash
/Users/starpoint/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/train_export.py
```

Then open:

```text
viewer/index.html
```

The generated data is written to `viewer/data/trajectory.js`, so the HTML file can be opened directly without a dev server.

## Model

The default network is:

```text
h_0 = x
h_l = |a_l h_{l-1} + b_l|, l = 1..L
y = c h_L + d
```

The colors stay attached to the original input coordinate. As training progresses, the same colored material points move through the learned sequence of affine stretches and absolute-value folds.

The default target is `sinmix`, a smooth oscillatory function. Use `--target abs` when you want the simplest `f(x) = |x|` fold.

## Useful Options

```bash
python3 scripts/train_export.py --target triangle --layers 6 --steps 250 --snapshots 251
python3 scripts/train_export.py --target sinmix --lr 0.01 --samples 700
python3 scripts/train_export.py --target abs --layers 6
python3 scripts/train_export.py --snapshot-schedule linear
python3 scripts/train_export.py --snapshot-power 3.0
python3 scripts/train_export.py --out viewer/data/trajectory.js
```

The default snapshot schedule is linear with one saved frame per training step from 0 to 250. Use `--snapshot-schedule early` when you want a smaller export that concentrates frames near the start.

If PyTorch is installed, you can request it explicitly:

```bash
python3 scripts/train_export.py --backend torch
```

The `auto` backend uses PyTorch when available and otherwise falls back to the built-in NumPy trainer.
