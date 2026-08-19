"""Apply dynamic INT8 quantization to the exported ONNX embedding model.

Dynamic quantization stores the model weights as INT8 and quantizes activations
on the fly at inference time. It needs **no calibration dataset**, which makes it a
clean, data-free baseline for the fp32 -> int8 speed/quality comparison.

Usage:
    uv run python scripts/export_onnx.py     # produces the fp32 ONNX export first
    uv run python scripts/quantize_onnx.py

Input:  models/minilm-onnx/onnx/model.onnx        (base export from scripts/export_onnx.py)
Output: models/minilm-onnx/onnx/model_int8.onnx

Serve the quantized model:
    ONNX_FILE_NAME=onnx/model_int8.onnx BACKEND=onnx uv run uvicorn app.main:app

Measure quality vs fp32 (cosine drift + nfcorpus nDCG):
    uv run python scripts/eval_quality.py --qa
"""
import tempfile
from contextlib import chdir
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from app.config import settings

ONNX_DIR = Path(settings.onnx_model_path) / "onnx"
SOURCE = ONNX_DIR / "model.onnx"
TARGET = ONNX_DIR / "model_int8.onnx"


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(
            f"{SOURCE} not found. Run `uv run python scripts/export_onnx.py` first."
        )

    source = SOURCE.resolve()
    target = TARGET.resolve()

    # quant_pre_process litters the cwd with sym_shape_infer_temp.onnx + external-data files,
    # so run it inside a temp dir (cwd) that is auto-removed. Paths are absolute to survive chdir.
    with tempfile.TemporaryDirectory() as tmp, chdir(tmp):
        quant_input = source
        preprocessed = Path(tmp) / "model_preprocessed.onnx"
        # Best-effort shape inference + graph cleanup. Symbolic shape inference can fail on some
        # exports; dynamic weight quantization doesn't require it, so fall back to the source.
        try:
            quant_pre_process(str(source), str(preprocessed), auto_merge=True)
            quant_input = preprocessed
        except Exception as exc:  # noqa: BLE001 - preprocessing is an optional optimization
            print(f"shape-inference preprocessing skipped ({exc}); quantizing source directly")

        # Dynamic quantization: INT8 weights, activations quantized at runtime. No calibration data.
        quantize_dynamic(
            model_input=str(quant_input),
            model_output=str(target),
            weight_type=QuantType.QInt8,
        )

    src_mb = source.stat().st_size / 1e6
    out_mb = target.stat().st_size / 1e6
    print(f"fp32: {src_mb:.1f} MB -> int8: {out_mb:.1f} MB  ({out_mb / src_mb:.0%} of original size)")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
