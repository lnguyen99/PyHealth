# Contributors: Jenny Nguyen (lmn4@illinois.edu),
#   Jordan Alwan (jzalwan2@illinois.edu), Karthik Pullela (kpull2@illinois.edu)
"""Deep Cox Mixtures with the EM loop inside the model.

This class subclasses
:class:`pyhealth.models.DeepCoxMixturesScript` and adds
a :meth:`DeepCoxMixturesModel.fit` method that runs the whole EM
training loop itself: per minibatch it E-steps with
:meth:`compute_posteriors`, M-steps with :meth:`q_loss` and Adam, and
refits the Breslow baselines on the full training set every
``update_splines_after`` batches. The loop is a port of
auton-survival's ``train_dcm`` / ``train_step``, with one deliberate
change: when a baseline refit fails, the failure is logged as a
warning instead of being swallowed silently (the reference code's
bare ``try/except`` reuses stale splines without saying so).

Design notes: one object owns everything, and after
:meth:`fit` the model can be evaluated with ``forward`` /
``Trainer.evaluate`` directly. But ``forward`` still cannot be a
standalone training step — it computes the Q-loss under the *stored*
baselines and never refits them, so ``Trainer.train()`` would optimise
against frozen baselines and silently do the wrong thing. Train this
class through :class:`DeepCoxMixtures`, which sets
``fit_outside_trainer`` so ``Trainer.train()`` raises a ``TypeError``
pointing at ``model.fit`` instead.

Paper: https://proceedings.mlr.press/v149/nagpal21a.html
Reference code: https://github.com/autonlab/auton-survival
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from pyhealth.models.deep_cox_mixtures_script import (
    DeepCoxMixturesScript,
)

__all__ = ["DeepCoxMixturesModel"]

logger = logging.getLogger(__name__)


class DeepCoxMixturesModel(DeepCoxMixturesScript):
    """Deep Cox Mixtures with the EM loop inside the model.

    See the module docstring for the design. The
    network, baselines, E-step, Q-loss and prediction are inherited
    from :class:`DeepCoxMixturesScript`; this class only adds the
    training loop.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "features": [0.0, 1.0, -0.5],
        ...         "duration": [1.2],
        ...         "event": [1.0],
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "features": [1.0, 0.2, 0.4],
        ...         "duration": [2.0],
        ...         "event": [0.0],
        ...     },
        ...     {
        ...         "patient_id": "p2",
        ...         "features": [0.5, -1.0, 0.3],
        ...         "duration": [3.0],
        ...         "event": [1.0],
        ...     },
        ...     {
        ...         "patient_id": "p3",
        ...         "features": [-0.4, 0.7, 1.1],
        ...         "duration": [4.0],
        ...         "event": [0.0],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = DeepCoxMixturesModel(dataset, k=2, layers=[4])
        >>> model.k
        2

    References:
        Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K.
        Deep Cox Mixtures for Survival Regression. Proceedings of
        Machine Learning for Healthcare, 2021.
    """

    def _em_epoch(
        self,
        features: Tensor,
        durations: Tensor,
        events: Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        typ: str,
        update_splines_after: int,
        use_posteriors: bool,
        epoch: int,
    ) -> None:
        """Run one EM epoch over minibatches (ported from ``train_step``)."""
        generator = torch.Generator(device=features.device)
        generator.manual_seed(epoch)
        perm = torch.randperm(
            features.shape[0], generator=generator, device=features.device
        )
        features, durations, events = (
            features[perm],
            durations[perm],
            events[perm],
        )
        # The baseline refit follows the training assignment mode, so the
        # mode is switchable via `typ`: 'hard' refits with argmax assignment
        # and 'soft' with sampled-hard assignment. Only 'true-soft' maps to
        # 'soft', because it is a posterior-weighted loss (not an assignment
        # mode) and the reference refits its baselines with sampled-hard
        # assignment. Note this differs from the reference script, which
        # hardcodes typ='soft' for the refit even when the M-step uses 'hard'.
        baseline_typ = "soft" if typ == "true-soft" else typ
        n_batches = (features.shape[0] // batch_size) + 1
        for i in range(n_batches):
            batch = slice(i * batch_size, (i + 1) * batch_size)
            batch_features = features[batch]
            batch_durations = durations[batch]
            batch_events = events[batch]
            if batch_features.shape[0] == 0:
                continue
            # E-step (no grad) then M-step.
            posteriors = self.compute_posteriors(
                batch_features, batch_durations, batch_events
            )
            optimizer.zero_grad()
            if typ == "true-soft":
                loss = self.soft_q_loss(
                    batch_features, batch_durations, batch_events, posteriors
                )
            else:
                loss = self.q_loss(
                    batch_features,
                    batch_durations,
                    batch_events,
                    posteriors,
                    typ=typ,
                )
            loss.backward()
            optimizer.step()
            # Refit the baselines on the full training set.
            if i % update_splines_after == 0:
                try:
                    if use_posteriors:
                        full_posteriors = self.compute_posteriors(
                            features, durations, events
                        )
                        self.fit_baselines(
                            features,
                            durations,
                            events,
                            posteriors=full_posteriors,
                            typ=baseline_typ,
                        )
                    else:
                        self.fit_baselines(
                            features, durations, events, typ=baseline_typ
                        )
                except Exception as exc:  # noqa: BLE001 - keep training
                    logger.warning(
                        "Could not fit Breslow splines, reusing the previous ones: %s",
                        exc,
                    )

    def _val_q_loss(
        self,
        features: Tensor,
        durations: Tensor,
        events: Tensor,
        typ: str,
    ) -> float:
        """Mean per-patient Q-loss on held-out data (ported from ``test_step``)."""
        with torch.no_grad():
            posteriors = self.compute_posteriors(features, durations, events)
            if typ == "true-soft":
                loss = self.soft_q_loss(features, durations, events, posteriors)
            else:
                loss = self.q_loss(features, durations, events, posteriors, typ=typ)
        return float(loss) / features.shape[0]

    def fit(
        self,
        features: Tensor,
        durations: Tensor,
        events: Tensor,
        val_features: Tensor | None = None,
        val_durations: Tensor | None = None,
        val_events: Tensor | None = None,
        epochs: int = 50,
        patience: int = 3,
        batch_size: int = 256,
        lr: float = 1e-3,
        typ: str = "soft",
        update_splines_after: int = 10,
        use_posteriors: bool = True,
        seed: int = 0,
        return_losses: bool = False,
    ) -> DeepCoxMixturesModel | tuple[DeepCoxMixturesModel, list[float]]:
        """Run the EM training loop inside the model.

        Ported from auton-survival's ``train_dcm``. Early stopping
        watches the validation Q-loss with ``patience`` epochs; with
        no validation data the training data is used instead.

        Args:
            features: Dense training inputs, shape ``(n, input_dim)``.
            durations: Observed training times, shape ``(n,)``.
            events: Training event indicators (1 = event, 0 = censored).
            val_features: Validation inputs; ``None`` reuses the
                training data for early stopping.
            val_durations: Validation times.
            val_events: Validation event indicators.
            epochs: Maximum number of EM epochs.
            patience: Epochs to wait for validation improvement.
            batch_size: Minibatch size for the M-step.
            lr: Adam learning rate (paper: 1e-3).
            typ: ``'soft'`` (sampled assignment, the reference
                default), ``'hard'`` (argmax assignment), or
                ``'true-soft'`` (posterior-weighted loss, Ablation 2).
            update_splines_after: Refit baselines every this many
                minibatches.
            use_posteriors: If ``True``, refit baselines from the
                full-data E-step posteriors; otherwise from the gate.
            seed: Random seed for the initial shuffle.
            return_losses: If ``True``, also return the per-epoch
                validation Q-losses.

        Returns:
            The fitted model, or ``(model, val_losses)`` when
            ``return_losses`` is set.
        """
        if typ not in ("soft", "hard", "true-soft"):
            raise ValueError("typ must be 'soft', 'hard' or 'true-soft'")
        torch.manual_seed(seed)
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        durations = torch.as_tensor(
            durations, dtype=torch.float32, device=self.device
        ).reshape(-1)
        events = torch.as_tensor(
            events, dtype=torch.float32, device=self.device
        ).reshape(-1)
        if val_features is None:
            val_features, val_durations, val_events = (
                features,
                durations,
                events,
            )
        else:
            val_features = torch.as_tensor(
                val_features, dtype=torch.float32, device=self.device
            )
            val_durations = torch.as_tensor(
                val_durations, dtype=torch.float32, device=self.device
            ).reshape(-1)
            val_events = torch.as_tensor(
                val_events, dtype=torch.float32, device=self.device
            ).reshape(-1)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val = float("inf")
        wait = 0
        losses: list[float] = []
        for epoch in range(epochs):
            self._em_epoch(
                features,
                durations,
                events,
                optimizer,
                batch_size,
                typ,
                update_splines_after,
                use_posteriors,
                epoch,
            )
            val_loss = self._val_q_loss(val_features, val_durations, val_events, typ)
            losses.append(val_loss)
            if val_loss > best_val:
                wait += 1
            else:
                wait = 0
            if wait == patience:
                break
            best_val = val_loss
        if return_losses:
            return self, losses
        return self
