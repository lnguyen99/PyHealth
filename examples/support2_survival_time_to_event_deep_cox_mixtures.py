"""Train and evaluate Deep Cox Mixtures (DCM) on SUPPORT2.

    python examples/support2_survival_time_to_event_deep_cox_mixtures.py --root data/support2
    python examples/support2_survival_time_to_event_deep_cox_mixtures.py --synthetic --epochs 2
    python examples/support2_survival_time_to_event_deep_cox_mixtures.py --folds 5 --k 3

DCM (Nagpal et al., MLHC 2021) is a mixture of K Cox models with a shared
encoder: conditioned on latent group Z the proportional-hazards assumption
holds, and each group's baseline survival is a spline-interpolated Breslow
estimator. This script is the EM training driver: the model file
``pyhealth/models/deep_cox_mixtures.py`` is a pure network plus
helpers, and the EM loop lives here, ported from the authors' reference
implementation in auton-survival (``auton_survival/models/dcm/``).
Do NOT ``pip install auton-survival`` — it pins torch 1.13 / numpy 1.24 /
pandas 1.5 and will clash with PyHealth; the code is ported, not imported.

What it shows: the full pipeline — load SUPPORT2 through
``Support2Dataset`` + ``SurvivalTimeToEventSupport2`` (27 features, 8,873
complete-case patients), standardise features with a scaler fitted on the
training split only, run the EM loop, and report time-dependent C-index,
cumulative/dynamic AUC, IPCW Brier score and survival ECE at the 25th /
50th / 75th event-time percentiles.

Dataset: SUPPORT2 is public, no registration. The script downloads it from
UCI (https://archive.ics.uci.edu/static/public/880/data.csv) into ``--root``
when the CSV is missing. ``--synthetic`` skips the download entirely and
trains on random tensors, to check that the code path runs.

Defaults run one 70/15/15 train/val/test split (``--folds 1``) so a first
run is quick. Pass ``--folds 5`` for the paper's 5-fold cross-validation;
each fold trains on 4/5 of the patients (with a 90/10 train/val split
inside for early stopping) and tests on the remaining 1/5. Metrics are
averaged over folds. ``--k 1`` is the single-Cox control arm for
Ablation 1; ``--assignment true-soft`` switches the M-step to the new
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

    K=3 (default): C-index 0.6237, C-td 0.6485; at 257 days AUC 0.6629,
        Brier 0.2303, ECE 0.0404. Early stop at epoch 26 of 50.
    K=1 (single-Cox control, Ablation 1): C-index 0.6141, C-td 0.6189;
        at 257 days AUC 0.6607, Brier 0.2302, ECE 0.0406. Early stop
        at epoch 21 of 50.

So the mixture beats the single Cox model on discrimination (C-td
+0.030, C-index +0.010) while calibration at the 75th percentile is
essentially tied. Both sit below the paper's 5-fold numbers, as
expected for one untuned fold on the 27-feature PyHealth task (the
paper uses 44 features and 5-fold CV); the full comparison runs with
``--folds 5``.

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

from pyhealth.datasets import Support2Dataset, create_sample_dataset
from pyhealth.metrics.survival import (
    concordance_index,
    cumulative_dynamic_auc,
    ipcw_brier_score,
)
from pyhealth.metrics.survival_curve import concordance_td
from pyhealth.models import DeepCoxMixtures
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


def train_em(
    model: DeepCoxMixtures,
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
    smoothing_factor: float = 1e-4,
    seed: int = 42,
) -> tuple[DeepCoxMixtures, list[float], int]:
    """EM training loop, ported from auton-survival's ``train_dcm``.

    Per epoch, per batch: an E-step under ``no_grad`` (random posteriors
    on the first pass, when no baselines exist yet), then an M-step that
    minimises the Q-loss with Adam. Baselines are refit on the full
    training set every ``update_splines_after`` batches. One deliberate
    deviation: a failed refit logs a warning and reuses the old splines
    instead of the reference's silent bare ``except``. Early stopping
    watches the validation Q-loss per patient with ``patience`` epochs.

    ``assignment`` selects the M-step loss: ``"sampled-hard"`` is the
    reference default (auton-survival's ``typ='soft'`` — each patient is
    sampled into one group), ``"hard"`` assigns by argmax, and
    ``"true-soft"`` uses the posterior-weighted loss (Ablation 2, new
    code; see :meth:`DeepCoxMixtures.soft_q_loss`).

    Returns ``(model, val_losses, epochs_run)``. As in the reference,
    the model is returned as-is at early stopping; no checkpoint is
    restored.
    """
    if assignment not in ("sampled-hard", "hard", "true-soft"):
        raise ValueError(f"Unknown assignment {assignment!r}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = x_train.shape[0]
    val_losses: list[float] = []
    best_val = math.inf
    waited = 0
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        perm = np.random.RandomState(seed + epoch).permutation(n_train)
        for batch_index, start in enumerate(range(0, n_train, batch_size)):
            idx = torch.as_tensor(perm[start : start + batch_size])
            xb, tb, eb = x_train[idx], t_train[idx], e_train[idx]
            posteriors = model.compute_posteriors(xb, tb, eb)
            optimizer.zero_grad()
            if assignment == "true-soft":
                loss = model.soft_q_loss(xb, tb, eb, posteriors)
            else:
                loss = model.q_loss(
                    xb,
                    tb,
                    eb,
                    posteriors,
                    typ="hard" if assignment == "hard" else "soft",
                )
            loss.backward()
            optimizer.step()
            # The baseline refit follows the assignment argument, so the mode
            # is switchable via --assignment: 'hard' refits with argmax
            # assignment and 'soft' with sampled-hard assignment. This differs
            # from the reference script, which hardcodes typ='soft' for the
            # refit even when the M-step uses 'hard'.
            if batch_index % update_splines_after == 0:
                try:
                    full_post = model.compute_posteriors(x_train, t_train, e_train)
                    model.fit_baselines(
                        x_train,
                        t_train,
                        e_train,
                        posteriors=full_post,
                        smoothing_factor=smoothing_factor,
                        typ="hard" if assignment == "hard" else "soft",
                    )
                except Exception as exc:  # noqa: BLE001 - refit is best-effort
                    logger.warning(
                        "Could not refit baselines at epoch %d batch %d "
                        "(%s); reusing previous splines",
                        epoch,
                        batch_index,
                        exc,
                    )
        with torch.no_grad():
            val_post = model.compute_posteriors(x_val, t_val, e_val)
            if assignment == "true-soft":
                val_loss = model.soft_q_loss(x_val, t_val, e_val, val_post)
            else:
                val_loss = model.q_loss(
                    x_val,
                    t_val,
                    e_val,
                    val_post,
                    typ="hard" if assignment == "hard" else "soft",
                )
        val_per_patient = float(val_loss) / x_val.shape[0]
        val_losses.append(val_per_patient)
        if val_per_patient > best_val:
            waited += 1
        else:
            waited = 0
        if waited == patience:
            logger.info(
                "Early stop at epoch %d (val Q-loss %.4f)", epoch, val_per_patient
            )
            break
        best_val = val_per_patient
    return model, val_losses, epochs_run


@torch.no_grad()
def evaluate(
    model: DeepCoxMixtures,
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
    model = DeepCoxMixtures(
        dataset,
        input_dim=features.shape[1],
        k=args.k,
        layers=args.layers,
        gamma=args.gamma,
        smoothing_factor=args.smoothing_factor,
    )
    model, val_losses, epochs_run = train_em(
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
        smoothing_factor=args.smoothing_factor,
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
