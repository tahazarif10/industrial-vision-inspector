from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from industrial_vision_inspector.data import make_loaders, prepare_neu_cls
from industrial_vision_inspector.engine import classification_metrics
from industrial_vision_inspector.image_ops import compute_quality_metrics, preprocess_grayscale
from industrial_vision_inspector.inference import predict_image
from industrial_vision_inspector.models import build_model
from industrial_vision_inspector.synthetic import CLASSES, generate_dataset


def test_quality_metrics_detect_edges_and_contrast():
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (48, 48), 220, -1)
    metrics = compute_quality_metrics(image)
    assert (
        metrics.contrast_std > 50
        and metrics.edge_density > 0
        and metrics.sharpness_laplacian_var > 0
    )


def test_preprocess_is_deterministic_shape_preserving():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (80, 90), dtype=np.uint8)
    first = preprocess_grayscale(image)
    second = preprocess_grayscale(image)
    assert first.shape == image.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_synthetic_dataset_has_all_splits_and_classes(tmp_path: Path):
    root = generate_dataset(
        tmp_path / "data",
        train_per_class=2,
        val_per_class=1,
        test_per_class=1,
        seed=3,
        size=64,
    )
    for split, expected in (("train", 2), ("val", 1), ("test", 1)):
        for class_name in CLASSES:
            assert len(list((root / split / class_name).glob("*.png"))) == expected


def test_loaders_preserve_class_order(tmp_path: Path):
    root = generate_dataset(
        tmp_path / "data",
        train_per_class=2,
        val_per_class=1,
        test_per_class=1,
        seed=4,
        size=64,
    )
    loaders, classes = make_loaders(root, image_size=32, batch_size=4)
    inputs, targets = next(iter(loaders["train"]))
    assert classes == sorted(CLASSES)
    assert inputs.shape[1:] == (3, 32, 32)
    assert targets.ndim == 1


def test_tinycnn_output_shape():
    assert build_model("tinycnn", 6)(torch.zeros(2, 3, 32, 32)).shape == (2, 6)


def test_inference_checkpoint_roundtrip(tmp_path: Path):
    classes = ["a", "b"]
    model = build_model("tinycnn", 2)
    checkpoint = {
        "format_version": 1,
        "model_name": "tinycnn",
        "pretrained_requested": False,
        "classes": classes,
        "image_size": 32,
        "use_cv_preprocess": True,
        "state_dict": model.state_dict(),
    }
    checkpoint_path = tmp_path / "model.pt"
    torch.save(checkpoint, checkpoint_path)
    data = generate_dataset(
        tmp_path / "data",
        train_per_class=1,
        val_per_class=1,
        test_per_class=1,
        seed=8,
        size=64,
    )
    image = next((data / "test" / "scratches").glob("*.png"))
    result = predict_image(checkpoint_path, image)
    assert result["prediction"] in classes
    assert 0.0 <= result["confidence"] <= 1.0
    assert "sharpness_laplacian_var" in result["quality"]


def test_prepare_neu_creates_stratified_manifest(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    for prefix in ("Cr", "Sc"):
        for i in range(10):
            assert cv2.imwrite(
                str(source / f"{prefix}_{i}.bmp"),
                np.full((16, 16), 30 + i, dtype=np.uint8),
            )
    manifest = prepare_neu_cls(
        source,
        tmp_path / "prepared",
        seed=9,
        train_fraction=0.6,
        val_fraction=0.2,
    )
    frame = pd.read_csv(manifest)
    counts = frame.groupby(["class", "split"]).size().to_dict()
    assert set(frame["class"]) == {"crazing", "scratches"}
    assert frame["sha256"].str.len().eq(64).all()
    assert counts[("crazing", "train")] == 6
    assert counts[("scratches", "test")] == 2


def test_classification_metrics_include_probabilistic_calibration():
    y_true = [0, 1, 1, 0]
    probabilities = [
        [0.90, 0.10],
        [0.20, 0.80],
        [0.55, 0.45],
        [0.60, 0.40],
    ]

    metrics = classification_metrics(y_true, probabilities, ["ok", "defect"], calibration_bins=5)

    assert metrics["sample_count"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["top2_accuracy"] == 1.0
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
    assert metrics["negative_log_likelihood"] > 0.0
    assert metrics["brier_score"] > 0.0
    assert sum(item["count"] for item in metrics["calibration_bins"]) == 4


def test_classification_metrics_reject_malformed_probability_rows():
    with pytest.raises(ValueError, match="sum to 1"):
        classification_metrics(
            [0, 1],
            [[0.8, 0.3], [0.2, 0.8]],
            ["ok", "defect"],
        )
