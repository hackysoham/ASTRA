# Team Astra -- Mars HiRISE Unsupervised Anomaly Detection Pipeline
# NSSC 2026, IIT Kharagpur | 
"""
tests/verify_shapes.py -- Pass a dummy tensor through every model
                          and assert exact output shape (B, 1, 227, 227).

Usage:
    python tests/verify_shapes.py           (from project root e:\\ASTRA)
"""

import sys
import os

# Ensure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from src.models import AEv1, AEv2, AEv3, AEv4, AEv5, MODEL_REGISTRY


EXPECTED_SHAPE = (2, 1, 227, 227)


def verify_single(name: str, model_cls: type) -> None:
    """Instantiate *model_cls*, forward a dummy batch, assert output shape."""
    model = model_cls()
    model.eval()

    x = torch.randn(*EXPECTED_SHAPE)

    with torch.no_grad():
        output = model(x)

    # VAE models return (x_hat, mu, logvar)
    if isinstance(output, tuple):
        x_hat = output[0]
        mu = output[1]
        logvar = output[2]
        # Latent shape check
        assert mu.shape == (EXPECTED_SHAPE[0], model.latent_dim), (
            f"{name} mu shape mismatch: got {mu.shape}, "
            f"expected ({EXPECTED_SHAPE[0]}, {model.latent_dim})"
        )
        assert logvar.shape == mu.shape, (
            f"{name} logvar shape mismatch: {logvar.shape} vs mu {mu.shape}"
        )
    else:
        x_hat = output

    # ---- The critical assertion ----
    assert x_hat.shape == EXPECTED_SHAPE, (
        f"{name} OUTPUT SHAPE MISMATCH: got {x_hat.shape}, expected {EXPECTED_SHAPE}"
    )

    # Value range: sigmoid output should be in [0, 1]
    assert x_hat.min() >= 0.0, f"{name} output has negative values: {x_hat.min()}"
    assert x_hat.max() <= 1.0, f"{name} output exceeds 1.0: {x_hat.max()}"

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"  {name:4s} | output={x_hat.shape} | "
        f"latent={model.latent_dim:4d} | "
        f"params={n_params:>12,} | "
        f"VAE={str(model.is_vae):5s} | PASS"
    )


def main() -> None:
    print("=" * 72)
    print("  SHAPE VERIFICATION -- Team Astra Autoencoder Architectures")
    print("  Input tensor:  torch.randn(2, 1, 227, 227)")
    print("  Expected out:  (2, 1, 227, 227)")
    print("=" * 72)

    passed = 0
    failed = 0

    for name, cls in MODEL_REGISTRY.items():
        try:
            verify_single(name, cls)
            passed += 1
        except AssertionError as exc:
            print(f"  {name:4s} | FAIL -- {exc}")
            failed += 1
        except Exception as exc:
            print(f"  {name:4s} | ERROR -- {type(exc).__name__}: {exc}")
            failed += 1

    print("-" * 72)
    if failed == 0:
        print(f"  ALL {passed} MODELS PASSED -- exact (B, 1, 227, 227) verified.")
    else:
        print(f"  {passed} passed, {failed} FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
