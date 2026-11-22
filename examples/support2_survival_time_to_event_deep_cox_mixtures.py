"""Train and evaluate Deep Cox Mixtures (DCM) on SUPPORT2.

    python examples/support2_survival_time_to_event_deep_cox_mixtures.py --root data/support2
    python examples/support2_survival_time_to_event_deep_cox_mixtures.py --synthetic --epochs 2
    python examples/support2_survival_time_to_event_deep_cox_mixtures.py --folds 5 --k 3

DCM (Nagpal et al., MLHC 2021) is a mixture of K Cox models with a shared
encoder: conditioned on latent group Z the proportional-hazards assumption
holds, and each group's baseline survival is a spline-interpolated Breslow
estimator. This script trains the model through
``model.fit(train_loader, val_loader)``: the EM loop lives inside the
model (``DeepCoxMixturesModelEM.fit``, ported from the authors' reference
implementation in auton-survival, ``auton_survival/models/dcm/``), and
this script builds the dataloaders, standardises the features, and
evaluates. Do NOT ``pip install auton-survival`` — it pins torch 1.13 /
numpy 1.24 / pandas 1.5 and will clash with PyHealth; the code is
ported, not imported.

What it shows: the full Option 2 pipeline — load SUPPORT2 through
``Support2Dataset`` + ``SurvivalTimeToEventSupport2`` (27 features, 8,873
complete-case patients), standardise features with a scaler fitted on the
training split only, fit with ``model.fit()``, and report time-dependent
C-index, cumulative/dynamic AUC, IPCW Brier score and survival ECE at the
25th / 50th / 75th event-time percentiles.

Dataset: SUPPORT2 is public, no registration. The script downloads it from
UCI (https://archive.ics.uci.edu/static/public/880/data.csv) into ``--root``
when the CSV is missing. ``--synthetic`` skips the download entirely and
trains on random tensors, to check that the code path runs.

Defaults run one 70/15/15 train/val/test split (``--folds 1``) so a first
run is quick. Pass ``--folds 5`` for the paper's 5-fold cross-validation;
each fold trains on 4/5 of the patients (with a 90/10 train/val split
inside for early stopping) and tests on the remaining 1/5. Metrics are
averaged over folds. ``--k 1`` is the single-Cox control arm for
Ablation 1; ``--assignment true-soft`` switches the M-step to the
posterior-weighted loss for Ablation 2 (auton-survival's ``typ='soft'``
is sampled-hard assignment, not true soft).

Paper targets on SUPPORT at the 75th event-time percentile (252 days),
5-fold CV pooled + 100 bootstraps (Table 6 of the paper):
C-td 0.6753, AUC 0.7256, ECE 0.0256, Brier 0.2118. The paper's consistent
win is calibration (ECE), not discrimination — it ties DSM/RSF on
discrimination and loses SUPPORT 25th/50th to RSF.

Results (1 fold, 70/15/15 split, seed 42, local CPU, Nov 21 2026;
horizons are the training fold's event-time quantiles, 14 / 60 / 257
days vs the paper's 14 / 58 / 252):

    K=3 (default): C-index 0.6163, C-td 0.6436; at 257 days AUC 0.6602,
        Brier 0.2305, ECE 0.0438. Ran all 50 epochs (no early stop).
    K=1 (single-Cox control, Ablation 1): C-index 0.6126, C-td 0.6180;
        at 257 days AUC 0.6591, Brier 0.2307, ECE 0.0529. Early stop
        at epoch 20 of 50.

So the mixture beats the single Cox model on discrimination (C-td
+0.026, C-index +0.004) and on calibration at the 75th percentile
(ECE 0.0438 vs 0.0529). Both sit below the paper's 5-fold numbers, as
expected for one untuned fold on the 27-feature PyHealth task (the
paper uses 44 features and 5-fold CV); the full comparison is Phase 3
(``--folds 5``). These numbers reproduce the equivalence check that
``model.fit()`` matches the in-model EM loop exactly (K=3 C-td 0.6436,
final val Q-loss 4.6408).

Deliberate deviations from the reference code, all documented where they
occur: a failed baseline refit logs a warning and reuses the old splines
instead of failing silently (the reference has a bare ``try/except``);
survival ECE is implemented here from scratch (20 bins, Kaplan-Meier per
bin) because neither PyHealth nor auton-survival has one; the script is
CPU-only, as in the reference (``.numpy()`` calls in the E-step and the
Breslow fit have no device handling).

Results print as a table and are written to ``--output`` (JSON).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pyhealth.datasets import (
    Support2Dataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.metrics.survival import (
    concordance_index,
    cumulative_dynamic_auc,
    ipcw_brier_score,
)
from pyhealth.metrics.survival_curve import concordance_td
from pyhealth.models import DeepCoxMixturesFitEM
from pyhealth.tasks import SurvivalTimeToEventSupport2

SUPPORT_URL = "https://archive.ics.uci.edu/static/public/880/data.csv"
N_FEATURES = 27
# Paper targets on SUPPORT at the 75th event-time percentile (Table 6).
PAPER_TARGETS = {"c_td": 0.6753, "auc": 0.7256, "ece": 0.0256, "brier": 0.2118}
ECE_BINS = 20

logger = logging.getLogger(__name__)


def download_support2(root: Path) -> Path:
    """Download the SUPPORT2 CSV into ``root`` if it is not there yet."""
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "support2.csv"
    if not csv_path.exists():
        logger.info("Downloading SUPPORT2 from %s", SUPPORT_URL)
        with urlopen(SUPPORT_URL, timeout=60) as response:
            frame = pd.read_csv(response).rename(columns={"id": "sno"})
        frame.to_csv(csv_path, index=False)
    return csv_path


def load_support2_arrays(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SUPPORT2 through PyHealth, returning ``(X, durations, events)``.

    Features are the ``SurvivalTimeToEventSupport2`` representation (27
    features: 8 numerics + 19 drop-first one-hots, 8,873 complete-case
    patients). They are returned unscaled; scaling is fitted on the
    training split only, inside :func:`run_fold`.
    """
    csv_path = download_support2(root)
    frame = pd.read_csv(csv_path)
    if len(frame) != 9105:
        raise ValueError(f"Expected 9,105 SUPPORT2 rows, got {len(frame)}")
    dataset = Support2Dataset(root=str(root), tables=["support2"])
    samples = dataset.set_task(SurvivalTimeToEventSupport2())
    if len(samples) != 8873:
        raise ValueError(f"Expected 8,873 samples, got {len(samples)}")
    features = np.array([np.asarray(s["features"], dtype=float) for s in samples])
    durations = np.array(
        [float(np.asarray(s["duration"]).reshape(-1)[0]) for s in samples]
    )
    events = np.array([float(np.asarray(s["event"]).reshape(-1)[0]) for s in samples])
    if features.shape != (8873, N_FEATURES):
        raise ValueError(f"Unexpected feature shape {features.shape}")
    return features, durations, events


def make_sample_dataset(
    features: np.ndarray, durations: np.ndarray, events: np.ndarray
):
    """Build a minimal PyHealth ``SampleDataset`` from arrays.

    >>> ds = make_sample_dataset(
    ...     np.zeros((2, 3)), np.array([1.0, 2.0]), np.array([1.0, 0.0])
    ... )
    >>> len(ds)
    2
    """
    samples = [
        {
            "patient_id": f"p{i}",
            "features": features[i].tolist(),
            "duration": [float(durations[i])],
            "event": [float(events[i])],
        }
        for i in range(len(durations))
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
    )


def kaplan_meier_at(durations: np.ndarray, events: np.ndarray, time: float) -> float:
    """Kaplan-Meier estimate of ``S(time)`` for one group of subjects.

    >>> round(kaplan_meier_at(
    ...     np.array([1.0, 2.0, 3.0]), np.array([1, 1, 0]), 2.0), 4)
    0.3333
    """
    durations = np.asarray(durations, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    survival = 1.0
    for event_time in np.unique(durations[events == 1]):
        if event_time > time:
            break
        at_risk = np.sum(durations >= event_time)
        died = np.sum((durations == event_time) & (events == 1))
        survival *= 1.0 - died / at_risk
    return float(survival)


def survival_ece(
    durations: np.ndarray,
    events: np.ndarray,
    surv_at_t: np.ndarray,
    time: float,
    n_bins: int = ECE_BINS,
) -> float:
    """Survival ECE at one horizon: mean predicted vs KM-observed survival.

    Subjects are binned by predicted ``S(time | x)`` into ``n_bins``
    equal-width bins over [0, 1]. Within each bin the observed survival
    is the Kaplan-Meier estimate at ``time``; the ECE is the
    size-weighted mean absolute gap between mean prediction and
    observation. This follows the paper's 20-bin ECE and is new code:
    neither PyHealth nor auton-survival implements a survival ECE.

    >>> round(survival_ece(
    ...     np.array([10.0, 10.0, 2.0, 2.0]), np.array([0, 0, 1, 1]),
    ...     np.array([0.9, 0.8, 0.2, 0.1]), 5.0, n_bins=2), 4)
    0.15
    """
    durations = np.asarray(durations, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=int).reshape(-1)
    surv_at_t = np.asarray(surv_at_t, dtype=float).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (surv_at_t >= lo) & (
            surv_at_t < hi if b < n_bins - 1 else surv_at_t <= hi
        )
        if not mask.any():
            continue
        observed = kaplan_meier_at(durations[mask], events[mask], time)
        total += mask.mean() * abs(float(surv_at_t[mask].mean()) - observed)
    return float(total)


def train_model(
    model: DeepCoxMixturesFitEM,
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    e_train: torch.Tensor,
    x_val: torch.Tensor,
    t_val: torch.Tensor,
    e_val: torch.Tensor,
    epochs: int = 50,
    patience: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    assignment: str = "sampled-hard",
    update_splines_after: int = 10,
    seed: int = 42,
) -> tuple[DeepCoxMixturesFitEM, list[float], int]:
    """Train through ``model.fit()`` on PyHealth dataloaders.

    The EM loop lives in the model; this function only builds
    unshuffled dataloaders from the split tensors (so the fit is
    reproducible) and maps ``assignment`` onto the model's ``typ``:
    ``"sampled-hard"`` -> ``"soft"`` (the reference default,
    auton-survival's ``typ='soft'`` — each patient is sampled into one
    group), ``"hard"`` -> ``"hard"`` (argmax assignment), and
    ``"true-soft"`` -> ``"true-soft"`` (posterior-weighted loss,
    Ablation 2). Returns ``(model, val_losses, epochs_run)``.
    """
    if assignment not in ("sampled-hard", "hard", "true-soft"):
        raise ValueError(f"Unknown assignment {assignment!r}")
    typ = {"sampled-hard": "soft", "hard": "hard", "true-soft": "true-soft"}[assignment]
    train_dataset = make_sample_dataset(
        x_train.numpy(), t_train.numpy(), e_train.numpy()
    )
    val_dataset = make_sample_dataset(x_val.numpy(), t_val.numpy(), e_val.numpy())
    train_loader = get_dataloader(train_dataset, batch_size=1024, shuffle=False)
    val_loader = get_dataloader(val_dataset, batch_size=1024, shuffle=False)
    model, val_losses = model.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        typ=typ,
        update_splines_after=update_splines_after,
        seed=seed,
        return_losses=True,
    )
    return model, val_losses, len(val_losses)


@torch.no_grad()
def evaluate(
    model: DeepCoxMixturesFitEM,
    x: torch.Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    horizons: list[float],
) -> dict[str, float]:
    """Score a fitted model: C-index, C-td, and per-horizon AUC/Brier/ECE.

    Risk scores are ``1 - S(t)`` (higher means an earlier event). The
    time-dependent concordance needs survival curves on a grid holding
    every event time, so curves are predicted at the unique observed
    times.
    """
    out: dict[str, float] = {}
    surv_by_horizon = {}
    for t in horizons:
        surv = model.predict_survival(x, torch.tensor([float(t)])).cpu().numpy()[:, 0]
        surv_by_horizon[float(t)] = surv
        risk = 1.0 - surv
        out[f"auc@{t:g}"] = float(
            cumulative_dynamic_auc(durations, events, risk, float(t))
        )
        out[f"brier@{t:g}"] = float(ipcw_brier_score(durations, events, surv, float(t)))
        out[f"ece@{t:g}"] = survival_ece(durations, events, surv, float(t))
    median_horizon = horizons[len(horizons) // 2]
    risk = 1.0 - surv_by_horizon[float(median_horizon)]
    out["c_index"] = float(concordance_index(durations, events, risk))
    grid = np.unique(durations)
    curves = model.predict_survival(x, torch.as_tensor(grid, dtype=torch.float32))
    out["c_td"] = float(concordance_td(durations, events, curves.cpu().numpy(), grid))
    return out


def run_fold(
    features: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Train and evaluate one fold; returns its metrics and horizons."""
    scaler = StandardScaler()
    x_train_raw = scaler.fit_transform(features[train_idx])
    x_val_raw = scaler.transform(features[val_idx])
    x_test_raw = scaler.transform(features[test_idx])
    to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)
    x_train, x_val, x_test = map(to_tensor, (x_train_raw, x_val_raw, x_test_raw))
    t_train = to_tensor(durations[train_idx])
    e_train = to_tensor(events[train_idx])
    t_val = to_tensor(durations[val_idx])
    e_val = to_tensor(events[val_idx])
    dataset = make_sample_dataset(x_train_raw, durations[train_idx], events[train_idx])
    model = DeepCoxMixturesFitEM(
        dataset,
        input_dim=features.shape[1],
        k=args.k,
        layers=args.layers,
        gamma=args.gamma,
        smoothing_factor=args.smoothing_factor,
    )
    model, val_losses, epochs_run = train_model(
        model,
        x_train,
        t_train,
        e_train,
        x_val,
        t_val,
        e_val,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        assignment=args.assignment,
        update_splines_after=args.update_splines_after,
        seed=args.seed,
    )
    train_events = durations[train_idx][events[train_idx] == 1]
    horizons = [float(q) for q in np.quantile(train_events, [0.25, 0.5, 0.75])]
    metrics = evaluate(model, x_test, durations[test_idx], events[test_idx], horizons)
    metrics["epochs_run"] = epochs_run
    metrics["final_val_q_loss"] = val_losses[-1] if val_losses else float("nan")
    return {"horizons": horizons, "metrics": metrics, "val_losses": val_losses}


def synthetic_arrays(
    n: int = 200, n_features: int = 8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small random dataset for ``--synthetic`` runs (no download)."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, n_features))
    durations = rng.exponential(scale=100.0, size=n) + 1.0
    events = rng.integers(0, 2, size=n).astype(float)
    return features, durations, events


def _jsonable(value):
    """Recursively replace NaN/inf with None so the JSON output is valid."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    """Parse arguments, run the folds, print and save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/support2", help="SUPPORT2 data dir")
    parser.add_argument(
        "--synthetic", action="store_true", help="random tensors, no download"
    )
    parser.add_argument(
        "--folds", type=int, default=1, help="CV folds (1 = one 70/15/15 split)"
    )
    parser.add_argument("--k", type=int, default=3, help="latent mixture groups")
    parser.add_argument(
        "--layers", type=int, nargs="*", default=[100], help="encoder hidden widths"
    )
    parser.add_argument(
        "--gamma", type=float, default=10.0, help="log hazard ratio clamp"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="minibatch size")
    parser.add_argument("--epochs", type=int, default=50, help="max EM epochs")
    parser.add_argument(
        "--patience", type=int, default=3, help="early-stop patience (epochs)"
    )
    parser.add_argument(
        "--assignment",
        default="sampled-hard",
        choices=["sampled-hard", "hard", "true-soft"],
        help="M-step assignment: reference sampled-hard, argmax hard, or true-soft",
    )
    parser.add_argument(
        "--update-splines-after",
        type=int,
        default=10,
        help="batches between baseline refits",
    )
    parser.add_argument(
        "--smoothing-factor",
        type=float,
        default=1e-4,
        help="baseline spline smoothing s",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--output", default="support2_dcm_results.json", help="results JSON path"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.synthetic:
        features, durations, events = synthetic_arrays(seed=args.seed)
    else:
        features, durations, events = load_support2_arrays(Path(args.root))
    n = len(durations)
    logger.info("Loaded %d patients, %d features", n, features.shape[1])

    splits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if args.folds <= 1:
        perm = np.random.RandomState(args.seed).permutation(n)
        n_test = int(0.15 * n)
        n_val = int(0.15 * n)
        splits.append(
            (
                perm[: n - n_test - n_val],
                perm[n - n_test - n_val : n - n_test],
                perm[n - n_test :],
            )
        )
    else:
        for fold, (train_all, test_idx) in enumerate(
            KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(
                np.arange(n)
            )
        ):
            inner = np.random.RandomState(args.seed + fold).permutation(train_all)
            n_val = max(1, int(0.10 * len(inner)))
            splits.append((inner[n_val:], inner[:n_val], test_idx))

    fold_results = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        logger.info(
            "Fold %d/%d: train %d, val %d, test %d",
            fold + 1,
            len(splits),
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )
        result = run_fold(
            features, durations, events, train_idx, val_idx, test_idx, args
        )
        fold_results.append(result)
        metrics = result["metrics"]
        logger.info(
            "Fold %d horizons (days): %s",
            fold + 1,
            [round(h, 1) for h in result["horizons"]],
        )
        logger.info(
            "Fold %d: c_index %.4f, c_td %.4f, epochs %d",
            fold + 1,
            metrics["c_index"],
            metrics["c_td"],
            metrics["epochs_run"],
        )
        for t in result["horizons"]:
            logger.info(
                "  t=%g days: auc %.4f, brier %.4f, ece %.4f",
                t,
                metrics[f"auc@{t:g}"],
                metrics[f"brier@{t:g}"],
                metrics[f"ece@{t:g}"],
            )

    mean_metrics: dict[str, float] = {}
    for key in fold_results[0]["metrics"]:
        values = [r["metrics"][key] for r in fold_results]
        mean_metrics[key] = float(np.mean(values))
    print(
        f"\nMean over {len(splits)} fold(s) "
        "(paper targets at 75th percentile in brackets):"
    )
    print(f"  c_index {mean_metrics['c_index']:.4f}")
    print(f"  c_td    {mean_metrics['c_td']:.4f}  [paper 0.6753]")
    horizons = fold_results[0]["horizons"]
    for t in horizons:
        print(
            f"  t={t:g} days: auc {mean_metrics[f'auc@{t:g}']:.4f} "
            f"[0.7256 at 75th], brier {mean_metrics[f'brier@{t:g}']:.4f} "
            f"[0.2118 at 75th], ece {mean_metrics[f'ece@{t:g}']:.4f} [0.0256 at 75th]"
        )
    results = {
        "config": {k: v for k, v in vars(args).items()},
        "paper_targets_75th": PAPER_TARGETS,
        "folds": fold_results,
        "mean_metrics": mean_metrics,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
