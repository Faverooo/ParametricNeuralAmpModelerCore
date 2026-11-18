"""Convert a ParametricNAM film_linear checkpoint and control profile to PNAM v1."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def export(checkpoint_path: Path, profile_path: Path, output_path: Path) -> None:
    numpy_scalars = [(np._core.multiarray.scalar, "numpy._core.multiarray.scalar"),
                     (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                     np.dtype, type(np.dtype("float32")), type(np.dtype("float64"))]
    with torch.serialization.safe_globals(numpy_scalars):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["format"] != "parametric_nam_a1" or config["variant"] != "film_linear":
        raise ValueError("Expected a ParametricNAM A1 film_linear checkpoint")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    controls = profile["controls"]
    if [control["id"] for control in controls] != config["knob_names"]:
        raise ValueError("Control IDs must match the checkpoint order")

    state = checkpoint["state_dict"]
    films = []
    for index in range(config["n_knob_layers"]):
        prefix = f"knob_projections.{index}."
        scale = state[prefix + "scale.weight"]
        shift = state[prefix + "shift.weight"]
        bias = state.get(prefix + "shift.bias", torch.zeros(scale.shape[0]))
        films.append({"scale": scale.tolist(), "shift": shift.tolist(), "bias": bias.tolist()})

    base = dict(checkpoint["base_nam"], sample_rate=config["sample_rate"])
    document = {
        "format": "pnam",
        "version": 1,
        "amplifier": profile["amplifier"],
        "controls": controls,
        "base_nam": base,
        "film_linear": films,
    }
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.profile, args.output)
