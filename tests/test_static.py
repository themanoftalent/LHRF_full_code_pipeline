from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = (ROOT / "matlab" / "run_lhrf.m").read_text(encoding="utf-8")

# Remove whitespace so formatting differences in the MATLAB file don't break tests
CODE_NOSPACE = "".join(CODE.split())


def test_key_components_present():
    """The main pipeline components are referenced in the MATLAB runner."""
    for token in (
        "detectSURFFeatures",
        "tripletProjection",
        "buildResidual",
        "linkage",
        "perfcurve",
    ):
        assert token in CODE, f"Missing component: {token}"


def test_image_size_and_relu_bottleneck():
    """Image resizing and post-ReLU SAE latent extraction match the manuscript."""
    assert "imresize(I,[224224])" in CODE_NOSPACE, "Image resize is not 224x224"
    assert "activations(ae,Xtr,'latent_relu'" in CODE_NOSPACE, \
        "SAE latent extraction is not post-ReLU"


def test_group_leakage_guard():
    """Group-leakage assertion is present."""
    assert "Groupleakagebetweentrainingandtest" in CODE_NOSPACE, \
        "Missing group-leakage guard"


def test_auc_class_order_guard():
    """Positive class is resolved by name, not by column index."""
    assert "posIndex" in CODE, "Missing positive-class index resolution"