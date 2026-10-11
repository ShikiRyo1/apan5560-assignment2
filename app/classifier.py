"""Load the bundled checkpoint once and classify decoded image bytes on CPU."""

import hashlib
from io import BytesIO
from pathlib import Path
from struct import error as StructError
import warnings

from PIL import Image, UnidentifiedImageError
import torch

from app.cnn import AssignmentCNN, CLASSES
from app.preprocessing import image_to_tensor

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "cifar10_cnn.pt"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class InvalidImageError(ValueError):
    """The file is not a supported, decodable image."""


class ImageLimitError(ValueError):
    """The file exceeds the byte or pixel limit."""


def decode_image(content: bytes) -> Image.Image:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ImageLimitError("Image file must be no larger than 5 MiB.")
    if not content:
        raise InvalidImageError("Upload a non-empty PNG or JPEG image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise InvalidImageError("Only PNG and JPEG images are supported.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ImageLimitError("Decoded image must have no more than 20 million pixels.")
                source.load()
                return source.copy()
    except (InvalidImageError, ImageLimitError):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageLimitError("Image dimensions exceed the decoding limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, StructError) as exc:
        raise InvalidImageError("The file could not be decoded as a PNG or JPEG image.") from exc


class ImageClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        torch.set_num_threads(4)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint["classes"] != list(CLASSES) or checkpoint["image_size"] != 64:
            raise RuntimeError("Checkpoint class mapping or input shape does not match this API.")
        if checkpoint["normalization_mean"] != [0.5] * 3 or checkpoint["normalization_std"] != [0.5] * 3:
            raise RuntimeError("Checkpoint preprocessing does not match this API.")
        self.model = AssignmentCNN()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.metadata = {
            "model": "AssignmentCNN", "dataset": "CIFAR-10", "device": "cpu",
            "input_shape": [1, 3, 64, 64], "classes": list(CLASSES),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "best_epoch": checkpoint["best_epoch"],
            "validation_accuracy": checkpoint["validation_accuracy"],
            "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "preprocessing": "EXIF orientation; RGB; bilinear resize 64x64; (x/255-0.5)/0.5",
            "score_note": "Softmax scores are relative model scores, not calibrated certainty.",
        }

    def classify(self, content: bytes) -> dict:
        image = decode_image(content)
        original_size = list(image.size)
        try:
            tensor = image_to_tensor(image)
        except (OSError, SyntaxError, ValueError, StructError) as exc:
            raise InvalidImageError("The image pixels or metadata could not be processed.") from exc
        with torch.inference_mode():
            scores = self.model(tensor).softmax(dim=1)[0]
        predicted_id = int(scores.argmax())
        return {
            "class_id": predicted_id,
            "class_name": CLASSES[predicted_id],
            "confidence": float(scores[predicted_id]),
            "scores": {name: float(score) for name, score in zip(CLASSES, scores)},
            "original_size": original_size,
            "input_shape": list(tensor.shape),
            "model": "AssignmentCNN",
        }
