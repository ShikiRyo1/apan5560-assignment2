"""Check the worksheet architecture, trained checkpoint and real HTTP inference."""

from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient

from app.cnn import AssignmentCNN, CLASSES
from app.classifier import MAX_UPLOAD_BYTES, ImageClassifier
from app.main import app
from app.preprocessing import image_to_tensor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_architecture_matches_assignment_and_backward_works():
    model = AssignmentCNN()
    leaves = [layer for layer in model.modules() if not list(layer.children())]
    assert [type(layer) for layer in leaves] == [
        nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Conv2d, nn.ReLU, nn.MaxPool2d,
        nn.Flatten, nn.Linear, nn.ReLU, nn.Linear,
    ]
    expected_shapes = [(2,16,64,64), (2,16,64,64), (2,16,32,32),
                       (2,32,32,32), (2,32,32,32), (2,32,16,16),
                       (2,8192), (2,100), (2,100), (2,10)]
    values = torch.rand(2, 3, 64, 64)
    for layer, shape in zip(leaves, expected_shapes):
        values = layer(values)
        assert tuple(values.shape) == shape
    assert sum(p.numel() for p in model.parameters()) == 825398
    loss = nn.CrossEntropyLoss()(values, torch.tensor([0, 9]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_preprocessing_known_pixel_values():
    black = image_to_tensor(Image.new("RGB", (32, 48), (0, 0, 0)))
    white = image_to_tensor(Image.new("RGB", (32, 48), (255, 255, 255)))
    assert black.shape == white.shape == (1, 3, 64, 64)
    assert torch.all(black == -1) and torch.all(white == 1)


def test_checkpoint_uses_validation_selected_epoch():
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    history = json.loads((ROOT / "results/history.json").read_text())
    best = max(history, key=lambda row: row["validation_accuracy"])
    service = ImageClassifier()
    assert service.metadata["best_epoch"] == best["epoch"] == metrics["best_epoch"]
    assert service.metadata["checkpoint_sha256"] == metrics["checkpoint_sha256"]
    assert not service.model.training


def test_split_is_exhaustive_disjoint_and_has_expected_counts():
    split = np.load(ROOT / "results/split_indices.npz")
    assert len(split["train"]) == 45000 and len(split["validation"]) == 5000
    assert len(set(split["train"]) & set(split["validation"])) == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([split["train"], split["validation"]])), np.arange(50000))
    assert metrics_counts() == (45000, 5000, 10000)


def metrics_counts():
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    return metrics["train_count"], metrics["validation_count"], metrics["test"]["count"]


@pytest.mark.parametrize("example", list(CLASSES))
def test_upload_matches_saved_test_prediction_and_direct_logits(client, example):
    content = (ROOT / f"examples/{example}.png").read_bytes()
    response = client.post("/classify", files={"file": (f"{example}.png", content, "image/png")})
    assert response.status_code == 200
    result = response.json()
    manifest = json.loads((ROOT / "examples/manifest.json").read_text())
    reference = next(row for row in manifest if row["true_class"] == example)
    assert result["class_name"] == reference["predicted_class"]
    assert result["class_name"] == CLASSES[result["class_id"]]
    assert result["input_shape"] == [1, 3, 64, 64]
    assert result["original_size"] == [32, 32]
    assert sum(result["scores"].values()) == pytest.approx(1.0, abs=1e-6)
    with Image.open(BytesIO(content)) as image, torch.inference_mode():
        expected = client.app.state.classifier.model(image_to_tensor(image)).softmax(1)[0]
    np.testing.assert_allclose(list(result["scores"].values()), expected.numpy(), atol=1e-6)


@pytest.mark.parametrize("mode,format_,content_type", [("L", "PNG", "image/png"), ("RGBA", "PNG", "image/png"), ("RGB", "JPEG", "image/jpeg")])
def test_non_rgb_and_non_square_supported_images(client, mode, format_, content_type):
    buffer = BytesIO()
    Image.new(mode, (47, 25)).save(buffer, format=format_)
    result = client.post("/classify", files={"file": ("test", buffer.getvalue(), content_type)})
    assert result.status_code == 200
    assert result.json()["original_size"] == [47, 25]


@pytest.mark.parametrize("content", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_empty_corrupt_and_falsely_labelled_uploads_are_rejected(client, content):
    assert client.post("/classify", files={"file": ("fake.png", content, "image/png")}).status_code == 422


def test_missing_upload_is_rejected(client):
    assert client.post("/classify").status_code == 422


def test_oversized_bytes_are_rejected(client):
    content = b"x" * (MAX_UPLOAD_BYTES + 1)
    assert client.post("/classify", files={"file": ("large.png", content, "image/png")}).status_code == 413


def test_oversized_decoded_image_is_rejected(client):
    buffer = BytesIO()
    Image.new("L", (5001, 4000)).save(buffer, format="PNG")
    assert client.post("/classify", files={"file": ("large.png", buffer.getvalue(), "image/png")}).status_code == 413


def test_unsupported_format_is_rejected(client):
    buffer = BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="GIF")
    assert client.post("/classify", files={"file": ("image.gif", buffer.getvalue(), "image/gif")}).status_code == 422


def test_model_metadata_and_openapi(client):
    result = client.get("/model")
    assert result.status_code == 200
    assert result.json()["classes"] == list(CLASSES)
    assert result.json()["parameter_count"] == 825398
    spec = client.get("/openapi.json").json()
    operation = spec["paths"]["/classify"]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]


@pytest.mark.parametrize("metadata_kind", ["zTXt", "iTXt", "iCCP"])
def test_compressed_metadata_limits_return_client_error(client, metadata_kind):
    # Small files can exceed Pillow's decompressed metadata limit before pixel loading.
    text = "x" * (PngImagePlugin.MAX_TEXT_CHUNK + 1)
    info = PngImagePlugin.PngInfo()
    kwargs = {}
    if metadata_kind == "zTXt":
        info.add_text("comment", text, zip=True)
    elif metadata_kind == "iTXt":
        info.add_itxt("comment", text, zip=True)
    else:
        kwargs["icc_profile"] = text.encode()
    buffer = BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG", pnginfo=info, **kwargs)
    content = buffer.getvalue()
    assert len(content) < 2048
    response = client.post("/classify", files={"file": ("metadata.png", content, "image/png")})
    assert response.status_code == 422
    assert "could not be decoded" in response.json()["detail"]


@pytest.mark.parametrize("raw_exif", ["zz", "49492a00"])
def test_malformed_exif_after_pixel_decode_returns_client_error(client, raw_exif):
    # Non-hex text raises ValueError; a truncated TIFF header raises struct.error.
    info = PngImagePlugin.PngInfo()
    info.add_text("Raw profile type exif", "header\nexif\nsize\n" + raw_exif)
    buffer = BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG", pnginfo=info)
    content = buffer.getvalue()
    with Image.open(BytesIO(content)) as source:
        source.load()  # The failure occurs later, during EXIF-aware preprocessing.
    response = client.post("/classify", files={"file": ("exif.png", content, "image/png")})
    assert response.status_code == 422
    assert "metadata could not be processed" in response.json()["detail"]


def test_model_fault_is_not_mislabeled_as_invalid_upload(client, monkeypatch):
    def fail_forward(*args, **kwargs):
        raise ValueError("simulated internal model fault")

    monkeypatch.setattr(client.app.state.classifier.model, "forward", fail_forward)
    content = (ROOT / "examples/cat.png").read_bytes()
    with pytest.raises(ValueError, match="simulated internal model fault"):
        client.post("/classify", files={"file": ("cat.png", content, "image/png")})
