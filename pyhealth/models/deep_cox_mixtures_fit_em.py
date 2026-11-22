# Contributors: Jenny Nguyen (lmn4@illinois.edu),
#   Jordan Alwan (jzalwan2@illinois.edu), Karthik Pullela (kpull2@illinois.edu)
"""Deep Cox Mixtures trained via ``fit_outside_trainer`` and
``model.fit()``.

This class subclasses
:class:`pyhealth.models.DeepCoxMixturesModelEM` and adds the
PyHealth mechanism for models that cannot be trained minibatch-wise:
the class attribute ``fit_outside_trainer = True`` makes
``Trainer.train()`` raise a ``TypeError`` directing the caller to
``model.fit(...)`` instead, and
:meth:`DeepCoxMixturesFitEM.fit` accepts PyHealth dataloaders,
collects their batches into full tensors, and runs the in-model EM
loop inherited from ``DeepCoxMixturesModelEM``. After fitting,
``Trainer.evaluate`` / inference / ``save_ckpt`` work as usual.

Why this shape: DCM's baselines are dataset-level (risk sets over
all patients), so no minibatch ``forward`` can train the model, and
PyHealth provides ``fit_outside_trainer`` exactly for this case —
``GradientBoostedTreeModel`` sets the same flag. The training entry
point is therefore PyHealth-shaped
(``model.fit(train_loader, val_loader)``) and the example script stays
thin. The cost is that the EM loop and the mutable spline state live
inside the model file, and ``forward`` still computes the Q-loss under
stored baselines, so ``Trainer.train()`` must stay off-limits.

Paper: https://proceedings.mlr.press/v149/nagpal21a.html
Reference code: https://github.com/autonlab/auton-survival
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from pyhealth.models.deep_cox_mixtures_model_em import (
    DeepCoxMixturesModelEM,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = ["DeepCoxMixturesFitEM"]


def _collect(loader: DataLoader) -> tuple[Tensor, Tensor, Tensor]:
    """Concatenate one dataloader's batches into full tensors.

    >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
    >>> dataset = create_sample_dataset(
    ...     samples=[{"patient_id": "p0", "features": [0.0, 1.0],
    ...               "duration": [1.0], "event": [1.0]},
    ...              {"patient_id": "p1", "features": [1.0, 0.2],
    ...               "duration": [2.0], "event": [0.0]}],
    ...     input_schema={"features": "tensor"},
    ...     output_schema={"duration": "tensor", "event": "tensor"},
    ... )
    >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    >>> features, durations, events = _collect(loader)
    >>> features.shape
    torch.Size([2, 2])
    """

    def _as_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
        return value[0] if isinstance(value, tuple) else value

    features_list, durations_list, events_list = [], [], []
    for batch in loader:
        features_list.append(_as_tensor(batch["features"]).float())
        durations_list.append(_as_tensor(batch["duration"]).float().reshape(-1))
        events_list.append(_as_tensor(batch["event"]).float().reshape(-1))
    return (
        torch.cat(features_list, dim=0),
        torch.cat(durations_list, dim=0),
        torch.cat(events_list, dim=0),
    )


class DeepCoxMixturesFitEM(DeepCoxMixturesModelEM):
    """Deep Cox Mixtures fitted via ``model.fit()``.

    See the module docstring for the design. The
    network, baselines, E-step, Q-loss, prediction and the EM loop
    are inherited from :class:`DeepCoxMixturesModelEM`; this class
    only adds the ``fit_outside_trainer`` flag and the dataloader
    entry point.

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
        >>> model = DeepCoxMixturesFitEM(dataset, k=2, layers=[4])
        >>> model.fit_outside_trainer
        True

    References:
        Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K.
        Deep Cox Mixtures for Survival Regression. Proceedings of
        Machine Learning for Healthcare, 2021.
    """

    #: Tells ``Trainer.train`` that this model is fit with ``model.fit``.
    fit_outside_trainer = True

    def fit(
        self,
        train_data: DataLoader,
        val_data: DataLoader | None = None,
        epochs: int = 50,
        patience: int = 3,
        batch_size: int = 256,
        lr: float = 1e-3,
        typ: str = "soft",
        update_splines_after: int = 10,
        use_posteriors: bool = True,
        seed: int = 0,
        return_losses: bool = False,
    ) -> DeepCoxMixturesFitEM | tuple[DeepCoxMixturesFitEM, list[float]]:
        """Fit the model on PyHealth dataloaders.

        Batches are concatenated in loader order and no sample is
        dropped, following the ``GradientBoostedTreeModel.fit``
        convention. Pass unshuffled loaders so the fit is
        reproducible. All other arguments are the EM-loop settings
        of :meth:`DeepCoxMixturesModelEM.fit`.

        Args:
            train_data: Training dataloader (the training split only).
            val_data: Optional validation dataloader, used for early
                stopping; ``None`` reuses the training data.
            epochs: Maximum number of EM epochs.
            patience: Epochs to wait for validation improvement.
            batch_size: Minibatch size for the M-step.
            lr: Adam learning rate (paper: 1e-3).
            typ: ``'soft'`` (sampled assignment) or ``'hard'``
                (argmax assignment).
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
        features, durations, events = _collect(train_data)
        val_tensors: tuple[Tensor, Tensor, Tensor] | None = None
        if val_data is not None:
            val_tensors = _collect(val_data)
        result = super().fit(
            features,
            durations,
            events,
            *(val_tensors if val_tensors is not None else (None, None, None)),
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            typ=typ,
            update_splines_after=update_splines_after,
            use_posteriors=use_posteriors,
            seed=seed,
            return_losses=return_losses,
        )
        return result
