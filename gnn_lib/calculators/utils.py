import urllib.request
from pathlib import Path


def get_cached_model(model_name: str, url: str) -> Path:
    """
    Download model checkpoint to cache if not present.

    Returns path to cached checkpoint.
    """
    cache_dir = Path.home() / ".cache" / "gnn_lib_models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cache_dir / f"{model_name}.pt"

    if not checkpoint_path.exists():
        print(f"Downloading pretrained model '{model_name}'...")
        print(f"From: {url}")
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"Saved to: {checkpoint_path}")
    else:
        print(f"Using cached model: {checkpoint_path}")
    return checkpoint_path