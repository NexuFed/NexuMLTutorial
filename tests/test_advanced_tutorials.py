"""Offline checks for tutorial-owned composition; real data is not downloaded."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.components import EvalBuildContext
from nexuml.core.config import ResolvedConfig
from nexuml.core.export import (
    export_package,
    export_safetensors,
    infer,
    load_package,
    load_inference_package,
    load_weights,
)
from nexuml.core.factory import callback
from nexuml.core.types import EvalAlgorithmSpec, LoaderSpec
from nexuml.data.dataset import NexuDataset
from nexuml.data.export.runner import export_data_module
from nexuml.data.exported import ExportedDataset
from nexuml.data.loaders.definitions import TorchLoader
from nexuml.data.module import NexuDataModule
from nexuml.training.lightning import (
    LightningFeatureExtractor,
    NexuLightningModule,
    NexuSession,
)
from tensordict import TensorDict
from torch.utils.data import Dataset

from examples.checkpoints_and_transfer import resume, transfer
from examples.export_and_infer import export_and_infer
from examples.prepare_speech_commands import prepare
from library.config.scenario import (
    mnist_resnet,
    fashion_mnist_transfer,
    speech_commands_log_mel,
    speech_commands_log_mel_prepared,
)
from library.config.tune import mnist_resnet as tuning

from library.data.mnist import FashionMNISTDataset, MNISTDataset
from library.evaluation import ConfusionMatrix


class _Images(Dataset):
    """A tiny torchvision-shaped payload, never a replacement runtime."""

    def __init__(self, root, train, download, transform):
        self.targets = torch.arange(20 if train else 10) % 10
        self.images = torch.rand(len(self.targets), 1, 28, 28)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


def _image_spec(monkeypatch, tmp_path, max_epochs=1):
    import torchvision

    monkeypatch.setattr(torchvision.datasets, "MNIST", _Images)
    monkeypatch.setattr(torchvision.datasets, "FashionMNIST", _Images)
    seed_everything(42)
    spec = mnist_resnet(
        encoder_width=8, encoder_depth=1, max_epochs=max_epochs, batch_size=4
    )
    spec.data.loader.num_workers = 0
    spec.data.train_split, spec.data.val_split, spec.data.test_split = 0.8, 0.2, 0.0
    spec.evaluation.algorithms = []
    spec.logging = None
    spec.exports = []
    spec.callbacks = [
        callback(ModelCheckpoint, dirpath=str(tmp_path / "checkpoints"), save_last=True)
    ]
    return spec


def test_tuning_trials_never_include_or_execute_test(monkeypatch):
    calls = []

    class Session:
        trainer_loggers = []

        @classmethod
        def from_scenario(cls, spec):
            assert len(spec.data.datasets) == 1
            assert spec.data.datasets[0].source.train is True
            assert spec.data.val_split == 0.2 and spec.data.test_split == 0
            assert not spec.evaluation.algorithms and not spec.exports
            calls.append(spec.name)
            return cls()

        def setup(self):
            return self

        def fit(self):
            return self

        def validate(self):
            return [{"val/loss": 0.5}]

        def test(self):
            pytest.fail("test must never be executed inside an Optuna trial")

    monkeypatch.setattr(tuning, "NexuSession", Session)
    for number in range(2):
        trial = SimpleNamespace(
            number=number,
            suggest_float=lambda *a, **kw: 1e-3,
            suggest_categorical=lambda name, choices: choices[0],
        )
        assert tuning.objective(trial, 1, "offline") == 0.5
    assert calls == ["offline_trial_0", "offline_trial_1"]
    selected = tuning.scenario(include_test=True)
    assert selected.data.datasets[1].source.train is False
    assert selected.data.datasets[1].split_type == "test"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_tuning_rejects_unavailable_objective(monkeypatch, value):
    monkeypatch.setattr(tuning.NexuSession, "setup", lambda self: self)
    monkeypatch.setattr(tuning.NexuSession, "fit", lambda self: self)
    monkeypatch.setattr(
        tuning.NexuSession,
        "validate",
        lambda self: [{}] if value is None else [{"val/loss": value}],
    )
    monkeypatch.setattr(
        tuning.NexuSession, "trainer_loggers", property(lambda self: [])
    )
    trial = SimpleNamespace(
        number=0,
        suggest_float=lambda *a, **k: 1e-3,
        suggest_categorical=lambda n, c: c[0],
    )
    with pytest.raises(ValueError, match="val/loss|not finite"):
        tuning.objective(trial, 1, "offline")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_log_mel_shape_device_and_singleton_parity(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pipeline = (
        compile_pipeline(speech_commands_log_mel(download=False)).to(device).eval()
    )
    transform = next(
        layer for stage, _, layer in pipeline.iter_layers() if stage == "Preprocessing"
    )
    waveform = torch.stack([torch.zeros(16_000), torch.randn(16_000)]).to(device)
    features = transform.forward_tensor(waveform)
    assert features.shape == (2, 1, 64, 101)
    assert features.device.type == device and torch.isfinite(features).all()
    torch.testing.assert_close(features[1:2], transform.forward_tensor(waveform[1:2]))
    out, _ = pipeline(TensorDict({"waveform": waveform}, batch_size=[2]))
    assert out["class_logits"].shape == (2, 8)


@pytest.mark.parametrize("backend", ["numpy", "webdataset"])
def test_prepared_round_trip_and_skip(tmp_path, backend):
    seed_everything(42)
    spec = speech_commands_log_mel(download=False)
    pipeline = compile_pipeline(spec).eval()
    waveform = torch.randn(6, 16_000)
    metadata = pd.DataFrame(
        {
            "speaker": ["a", "a", "b", "b", "c", "c"],
            "identity": list(range(6)),
            "class": [0, 1, 2, 3, 4, 5],
            "split": ["train", "train", "val", "val", "test", "test"],
        }
    )
    dataset = NexuDataset(
        data=[TensorDict({"waveform": w}, batch_size=[]) for w in waveform],
        meta=metadata,
        label_names=["class"],
        modality="audio",
    )
    module = NexuDataModule(
        dataset, LoaderSpec(backend=TorchLoader(), batch_size=2), split_by_column=True
    )
    path = export_data_module(
        module,
        tmp_path / backend,
        backend=backend,
        transform=LightningFeatureExtractor(
            NexuLightningModule(pipeline), x_keys=["features"]
        ),
        x_keys=["features"],
        y_keys=["class"],
        device="cpu",
    )
    prepared = ExportedDataset(path)
    assert len(prepared) == 6
    expected = LightningFeatureExtractor(
        NexuLightningModule(pipeline), x_keys=["features"]
    )(
        TensorDict({"waveform": waveform}, batch_size=[6]),
    )[0]["features"]
    for i, row in prepared.meta.iterrows():
        x, y = prepared[i]
        identity = int(row["identity"])
        torch.testing.assert_close(x["features"], expected[identity])
        assert y["class"].item() == metadata.loc[identity, "class"]
        assert row["split"] == metadata.loc[identity, "split"]
    target = speech_commands_log_mel_prepared(root=str(path))
    assert ResolvedConfig.from_yaml(
        ResolvedConfig.from_scenario(target).to_yaml()
    ).data.skip_pipeline_stages == ["Preprocessing"]
    compiled = compile_pipeline(target).eval()
    assert all(stage != "Preprocessing" for stage, _, _ in compiled.iter_layers())
    # Compare the reused downstream layers, not random initializations.
    source_layers = [
        layer for stage, _, layer in pipeline.iter_layers() if stage != "Preprocessing"
    ]
    for source, (_, _, layer) in zip(
        source_layers, compiled.iter_layers(), strict=True
    ):
        layer.load_state_dict(source.state_dict())
    pipeline.eval()
    online, _ = pipeline(TensorDict({"waveform": waveform}, batch_size=[6]))
    offline, _ = compiled(TensorDict({"features": expected}, batch_size=[6]))
    torch.testing.assert_close(online["class_logits"], offline["class_logits"])


def test_confusion_counts_reset_and_artifact(tmp_path):
    from lightning.pytorch.loggers import TensorBoardLogger
    from nexuml.evaluation.algorithm import ContractError

    evaluator = ConfusionMatrix(class_names=["a", "b", "absent"]).build(
        EvalBuildContext()
    )
    x = TensorDict(
        {
            "class_logits": torch.tensor(
                [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
            )
        },
        batch_size=[3],
    )
    y = TensorDict({"class": torch.tensor([0, 1, 1])}, batch_size=[3])
    evaluator.fit_batch(x, y)
    with pytest.raises(ValueError, match="no samples"):
        evaluator.eval_end()
    evaluator.eval_batch(x, y)
    evaluator.eval_end()
    assert evaluator.results() == {"micro_accuracy": 2 / 3, "samples": 3.0}
    assert evaluator.counts.compute().tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
    logger = TensorBoardLogger(str(tmp_path), name="matrix")
    evaluator.visualize(logger)
    logger.finalize("success")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(logger.log_dir).Reload()
    assert "eval/confusion_matrix" in events.Tags()["images"]
    evaluator.eval_batch(x, y)
    evaluator.eval_end()
    assert evaluator.results()["samples"] == 3  # not 6 on a repeated evaluation
    with pytest.raises(ContractError):
        evaluator.eval_batch(TensorDict({}, batch_size=[3]), y)


def test_transfer_no_matches_and_export_destination_guard(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    session = SimpleNamespace(runtime=SimpleNamespace(load_report={"matched": []}))
    session.setup = lambda: session
    monkeypatch.setattr(NexuSession, "from_scenario", lambda spec, **kwargs: session)
    with pytest.raises(ValueError, match="No encoder weights matched"):
        transfer(source, tmp_path / "target")
    with pytest.raises(FileExistsError):
        prepare("missing", source, "numpy", "cpu")


def test_confusion_public_session_results(monkeypatch, tmp_path):
    spec = _image_spec(monkeypatch, tmp_path)
    spec.evaluation.algorithms = [
        EvalAlgorithmSpec(
            algorithm=ConfusionMatrix(class_names=list(map(str, range(10)))),
            feature_key="class_logits",
            label_key="class",
        )
    ]
    session = NexuSession.from_scenario(
        spec, accelerator="cpu", enable_loggers=False, enable_progress_bar=False
    ).fit()
    session.test()
    assert (
        session.lightning_module.evaluation_results["TutorialConfusionMatrix/samples"]
        == 10
    )
    session.test()
    assert (
        session.lightning_module.evaluation_results["TutorialConfusionMatrix/samples"]
        == 10
    )


def test_transfer_dataset_and_selective_freeze(monkeypatch, tmp_path):
    spec = _image_spec(monkeypatch, tmp_path)
    assert MNISTDataset(download=False).build()[0][0]["features"].shape == (1, 28, 28)
    assert len(FashionMNISTDataset(train=True).build()) == 20
    assert len(FashionMNISTDataset(train=False).build()) == 10
    source = compile_pipeline(spec)
    artifact = export_safetensors(source, tmp_path / "source.safetensors")
    target_spec = fashion_mnist_transfer(
        source=str(artifact), encoder_width=8, encoder_depth=1
    )
    target = compile_pipeline(target_spec)
    initial_head = {
        k: v.clone()
        for k, v in target.state_dict().items()
        if k.startswith("stages.Head.")
    }
    report = load_weights(target, artifact, checkpoint=target_spec.checkpoint)
    assert report.matched and all(
        k.startswith("stages.Encoder.") for k in report.matched
    )
    for key in report.matched:
        torch.testing.assert_close(target.state_dict()[key], source.state_dict()[key])
    for key, value in initial_head.items():
        torch.testing.assert_close(target.state_dict()[key], value)
    encoder_before = {
        k: v.detach().clone()
        for k, v in target.named_parameters()
        if k.startswith("stages.Encoder.")
    }
    target.train()
    optimizer = target.create_optimizer()
    x, _ = target(
        TensorDict({"features": torch.rand(4, 1, 28, 28)}, batch_size=[4]),
        TensorDict({"class": torch.tensor([0, 1, 2, 3])}, batch_size=[4]),
    )
    x["classification_loss"].mean().backward()
    optimizer.step()
    assert all(
        not p.requires_grad
        for k, p in target.named_parameters()
        if k.startswith("stages.Encoder.")
    )
    for k, p in target.named_parameters():
        if k in encoder_before:
            torch.testing.assert_close(p, encoder_before[k])
    assert any(
        not torch.equal(target.state_dict()[k], v) for k, v in initial_head.items()
    )
    fine_artifact = export_safetensors(target, tmp_path / "target.safetensors")
    fine = compile_pipeline(target_spec)
    load_weights(
        fine,
        fine_artifact,
        allow_missing=False,
        allow_shape_mismatch=False,
        freeze_loaded=False,
    )
    assert all(p.requires_grad for p in fine.parameters())
    incompatible = compile_pipeline(
        fashion_mnist_transfer(encoder_width=16, encoder_depth=1)
    )
    with pytest.raises(ValueError, match="[Ss]hape"):
        load_weights(incompatible, artifact, checkpoint=target_spec.checkpoint)
    with pytest.raises(FileNotFoundError):
        load_weights(target, tmp_path / "missing.safetensors")


def test_resume_recovers_optimizer_and_advances(monkeypatch, tmp_path):
    spec = _image_spec(monkeypatch, tmp_path)
    first = NexuSession.from_scenario(
        spec, accelerator="cpu", enable_loggers=False, enable_progress_bar=False
    ).fit()
    saved = Path(first.trainer.checkpoint_callback.last_model_path)
    checkpoint = torch.load(saved, weights_only=False)
    resumed = NexuSession.from_trainer_checkpoint(
        saved, accelerator="cpu", enable_loggers=False, enable_progress_bar=False
    )
    resumed.scenario.training.max_epochs = 2
    resumed.setup()
    for k, v in checkpoint["state_dict"].items():
        torch.testing.assert_close(resumed.lightning_module.state_dict()[k], v)
    captured = []

    class CheckRestore(Callback):
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == checkpoint["global_step"]
            actual = trainer.optimizers[0].state_dict()
            expected = checkpoint["optimizer_states"][0]
            for key, state in expected["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(actual["state"][key][name], value)
            assert (
                trainer.lr_scheduler_configs[0].scheduler.state_dict()
                == checkpoint["lr_schedulers"][0]
            )
            captured.append(True)

    resumed.trainer.callbacks.append(CheckRestore())
    resumed.fit()
    assert captured and resumed.trainer.global_step > checkpoint["global_step"]
    assert resumed.trainer.current_epoch == 2
    with pytest.raises(FileNotFoundError):
        resume(tmp_path / "missing.ckpt", 2)


def test_package_safetensors_and_onnx_parity(monkeypatch, tmp_path):
    spec = _image_spec(monkeypatch, tmp_path)
    session = NexuSession.from_scenario(
        spec, accelerator="cpu", enable_loggers=False, enable_progress_bar=False
    ).fit()
    package = export_package(
        session.pipeline, tmp_path / "model", trainer=session.trainer
    )
    batch = TensorDict({"features": torch.rand(2, 1, 28, 28)}, batch_size=[2])
    expected = infer(session.pipeline, batch.clone())["class_logits"]
    for loader in (load_package, load_inference_package):
        restored, _, _ = loader(package)
        torch.testing.assert_close(
            infer(restored, batch.clone())["class_logits"], expected
        )
    export_and_infer(package, tmp_path / "portable")
    with pytest.raises(FileExistsError):
        export_and_infer(package, tmp_path / "portable")
    with pytest.raises(FileNotFoundError):
        export_and_infer(tmp_path / "missing", tmp_path / "unused")
    pytest.importorskip("onnxruntime", reason="install .[onnx] for runtime parity")
    pytest.importorskip("onnxscript", reason="install .[onnx] for exporter")
    export_and_infer(package, tmp_path / "onnx", onnx=True)
