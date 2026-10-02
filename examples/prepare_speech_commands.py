"""Export log-mel features with NexuML, keeping labels and speaker-owned splits."""

import argparse
from pathlib import Path

from nexuml.data.export.runner import export_data_module
from nexuml.training.lightning import LightningFeatureExtractor, NexuSession

from library.config.scenario import speech_commands_log_mel


def prepare(
    root: str, output: Path, backend: str, device: str, batch_size: int = 64
) -> Path:
    if output.exists():
        raise FileExistsError(
            f"Choose a new export destination; {output} already exists"
        )
    spec = speech_commands_log_mel(root=root, batch_size=batch_size, num_workers=0)
    session = NexuSession.from_scenario(spec)
    return export_data_module(
        session.data_module,
        output,
        backend=backend,
        transform=LightningFeatureExtractor(
            session.lightning_module, x_keys=["features"]
        ),
        x_keys=["features"],
        y_keys=["class"],
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/mini_speech_commands")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=["numpy", "webdataset"], default="webdataset"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(prepare(**vars(args)))


if __name__ == "__main__":
    main()
