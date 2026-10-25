"""Reload a trained MNIST package, export weights/ONNX, and check inference parity."""

import argparse
from pathlib import Path

import torch
from nexuml.core.compiler import compile as compile_pipeline
from nexuml.core.export import (
    export_onnx,
    export_safetensors,
    infer,
    load_inference_package,
    load_package,
    load_weights,
)
from tensordict import TensorDict


def export_and_infer(source: Path, output: Path, onnx: bool = False) -> None:
    if not source.is_dir():
        raise FileNotFoundError(
            f"Expected an explicit trained package directory: {source}"
        )
    output.mkdir(parents=True, exist_ok=False)
    pipeline, config, _ = load_package(source)
    packaged, _, _ = load_inference_package(source)
    shape = config.data.input_shapes["features"]
    if shape != [1, 28, 28]:
        raise ValueError("This example expects the MNIST image classifier [1, 28, 28]")
    images = torch.rand(2, *shape, generator=torch.Generator().manual_seed(42))
    batch = TensorDict({"features": images}, batch_size=[2])
    expected = infer(pipeline, batch.clone())["class_logits"]
    torch.testing.assert_close(
        expected, infer(packaged, batch.clone())["class_logits"], rtol=1e-4, atol=1e-5
    )

    weights = export_safetensors(pipeline, output / "weights.safetensors")
    matching = compile_pipeline(config.to_scenario())
    report = load_weights(
        matching, weights, allow_missing=False, allow_shape_mismatch=False
    )
    print("SafeTensors load report:", report.to_dict())
    torch.testing.assert_close(
        expected, infer(matching, batch.clone())["class_logits"], rtol=1e-4, atol=1e-5
    )
    print(
        "Package and SafeTensors parity passed; predicted classes:",
        expected.argmax(-1).tolist(),
    )

    if onnx:
        import onnxruntime as ort

        graph = export_onnx(
            pipeline,
            output / "model.onnx",
            input_key="features",
            output_key="class_logits",
        )
        runtime = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
        for size in (1, 2):
            inputs = images[:size]
            actual = runtime.run(["class_logits"], {"features": inputs.numpy()})[0]
            reference = infer(
                pipeline, TensorDict({"features": inputs}, batch_size=[size])
            )["class_logits"]
            torch.testing.assert_close(
                reference, torch.from_numpy(actual), rtol=1e-4, atol=1e-5
            )
            assert torch.equal(
                reference.argmax(-1), torch.from_numpy(actual).argmax(-1)
            )
        print("ONNX Runtime parity passed for batch sizes 1 and 2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", action="store_true")
    export_and_infer(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
