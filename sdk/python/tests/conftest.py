import json
from pathlib import Path

VECTORS = Path(__file__).parent / "vectors"

def load_vectors(name: str) -> dict:
    return json.loads((VECTORS / name).read_text())
