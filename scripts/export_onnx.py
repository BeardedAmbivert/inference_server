"""Export the SentenceTransformer model to optimized ONNX format.

Usage:
    uv run python scripts/export_onnx.py

This script:
1. Loads the SentenceTransformer model
2. Saves the base model to disk
3. Exports to optimized ONNX using sentence-transformers built-in export
4. Verifies the export by loading and running inference

Output: models/minilm-onnx/

Verification (after export):
    uv run python -c "
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('models/minilm-onnx', backend='onnx')
    emb = model.encode(['test sentence'])
    print(f'Shape: {emb.shape}  (expected: (1, 384))')
    "
"""
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.backend import export_optimized_onnx_model
from app.config import settings


onnx_model = SentenceTransformer(
    model_name_or_path=settings.model_name)

onnx_model.save_pretrained(settings.onnx_model_path)

export_optimized_onnx_model(
    onnx_model,
    optimization_config="O3",
    model_name_or_path=settings.onnx_model_path,
)

model_onnx = SentenceTransformer(settings.onnx_model_path, backend="onnx")
test_emb = model_onnx.encode(["hello world"])
print(f"Shape: {test_emb.shape}  (expected: (1, 384))")

print(os.listdir(settings.onnx_model_path))
