# Contributors: Jenny Nguyen (lmn4@illinois.edu),
#   Jordan Alwan (jzalwan2@illinois.edu), Karthik Pullela (kpull2@illinois.edu)
"""Fast synthetic tests for the in-model EM training of DCM.

Covers :class:`pyhealth.models.DeepCoxMixturesModelEM` (EM loop inside
the model) and
:class:`pyhealth.models.DeepCoxMixturesFitEM` (``fit_outside_trainer``
+ ``model.fit``). The base network class is covered by
``test_deep_cox_mixtures_script_em.py``. All data here is synthetic;
no real or demo dataset is used.
"""

import pytest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.deep_cox_mixtures_fit_em import DeepCoxMixturesFitEM
from pyhealth.models.deep_cox_mixtures_model_em import (
    DeepCoxMixturesModelEM,
)


def _make_dataset(n: int = 8):
    samples = [
        {
            "patient_id": f"p{i}",
            "features": [float(i % 3), float(i) / n, -0.5 + 0.1 * i],
            "duration": [1.0 + float(i)],
            "event": [1.0 if i % 2 == 0 else 0.0],
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="deep_cox_mixtures_variants_test",
    )


def _tensors(dataset):
    loader = get_dataloader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    def _as_tensor(value):
        return value[0] if isinstance(value, tuple) else value

    return (
        _as_tensor(batch["features"]).float(),
        _as_tensor(batch["duration"]).float().reshape(-1),
        _as_tensor(batch["event"]).float().reshape(-1),
    )


def test_model_em_fit_runs_and_predicts():
    dataset = _make_dataset()
    model = DeepCoxMixturesModelEM(
        dataset, k=2, layers=[4], time_grid=torch.linspace(1.0, 8.0, 5)
    )
    features, durations, events = _tensors(dataset)
    fitted, losses = model.fit(
        features,
        durations,
        events,
        epochs=3,
        patience=3,
        batch_size=4,
        return_losses=True,
    )
    assert fitted is model
    assert len(losses) == 3
    assert all(isinstance(loss, float) for loss in losses)
    assert model.breslow_splines is not None
    survival = model.predict_survival(features, torch.tensor([1.0, 4.0, 8.0]))
    assert survival.shape == (8, 3)
    assert torch.all(survival >= 0.0) and torch.all(survival <= 1.0)
    assert torch.all(survival[:, 1:] <= survival[:, :-1] + 1e-6)
    output = model(**next(iter(get_dataloader(dataset, batch_size=8, shuffle=False))))
    assert torch.isfinite(output["loss"])
    assert output["y_prob"].shape == (8, 5)


def test_model_em_fit_early_stops():
    dataset = _make_dataset()
    model = DeepCoxMixturesModelEM(dataset, k=2, layers=[4])
    features, durations, events = _tensors(dataset)
    _, losses = model.fit(
        features,
        durations,
        events,
        epochs=10,
        patience=1,
        batch_size=4,
        return_losses=True,
    )
    assert len(losses) <= 10


def test_fit_em_flag_and_dataloader_fit():
    dataset = _make_dataset()
    model = DeepCoxMixturesFitEM(
        dataset, k=2, layers=[4], time_grid=torch.linspace(1.0, 8.0, 5)
    )
    assert model.fit_outside_trainer is True
    train_loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    val_loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    fitted = model.fit(train_loader, val_loader, epochs=2, patience=2, batch_size=4)
    assert fitted is model
    assert model.breslow_splines is not None
    features, _, _ = _tensors(dataset)
    survival = model.predict_survival(features, torch.tensor([1.0, 4.0, 8.0]))
    assert survival.shape == (8, 3)
    assert torch.all(torch.isfinite(survival))


def test_fit_em_trainer_train_refuses():
    from pyhealth.trainer import Trainer

    dataset = _make_dataset()
    model = DeepCoxMixturesFitEM(dataset, k=2, layers=[4])
    trainer = Trainer(model)
    train_loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    with pytest.raises(TypeError, match="model.fit"):
        trainer.train(train_dataloader=train_loader)
