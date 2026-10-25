"""Resume a trainer checkpoint, or initialize and fine-tune a new task."""

import argparse
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from nexuml.core.export import export_package
from nexuml.core.factory import callback
from nexuml.core.types import CheckpointLoadSpec
from nexuml.training.lightning import NexuSession

from library.config.scenario import fashion_mnist_transfer


def resume(checkpoint: Path, max_epochs: int) -> NexuSession:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    session = NexuSession.from_trainer_checkpoint(checkpoint)
    # max_epochs is the total target, not a number of additional epochs.
    session.scenario.training.max_epochs = max_epochs
    session.fit()
    print(
        f"Resumed epoch={session.trainer.current_epoch}, step={session.trainer.global_step}"
    )
    return session


def transfer(source: Path, output: Path, max_epochs: int = 1) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    output.mkdir(parents=True, exist_ok=False)
    spec = fashion_mnist_transfer(source=str(source), max_epochs=max_epochs)
    spec.callbacks = [
        callback(
            ModelCheckpoint,
            dirpath=str(output / "head_checkpoints"),
            save_last=True,
            monitor="val/loss",
            mode="min",
        )
    ]
    head = NexuSession.from_scenario(spec, run_name="fashion-head").setup()
    report = head.runtime.load_report
    if not report or not report["matched"]:
        raise ValueError(
            "No encoder weights matched; check source architecture and include pattern"
        )
    print("Encoder load report:", report)
    # ponytail: frozen parameters still update BatchNorm buffers; use an eval backbone for fixed statistics.
    head.fit()
    head_package = export_package(head.pipeline, output / "head", trainer=head.trainer)

    fine_spec = fashion_mnist_transfer(max_epochs=max_epochs, lr=1e-4)
    fine_spec.checkpoint = CheckpointLoadSpec(
        source=str(head_package),
        allow_missing=False,
        allow_shape_mismatch=False,
        freeze_loaded=False,
    )
    fine_spec.callbacks = [
        callback(
            ModelCheckpoint,
            dirpath=str(output / "fine_checkpoints"),
            save_last=True,
            monitor="val/loss",
            mode="min",
        )
    ]
    fine = NexuSession.from_scenario(fine_spec, run_name="fashion-fine").setup().fit()
    print("Fine-tuned target test results:", fine.test())
    export_package(fine.pipeline, output / "fine", trainer=fine.trainer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resumed = commands.add_parser("resume")
    resumed.add_argument("--checkpoint", type=Path, required=True)
    resumed.add_argument("--max-epochs", type=int, required=True)
    transferred = commands.add_parser("transfer")
    transferred.add_argument("--source", type=Path, required=True)
    transferred.add_argument("--output", type=Path, required=True)
    transferred.add_argument("--max-epochs", type=int, default=1)
    args = vars(parser.parse_args())
    command = args.pop("command")
    (resume if command == "resume" else transfer)(**args)


if __name__ == "__main__":
    main()
