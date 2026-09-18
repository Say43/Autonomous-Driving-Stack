#!/usr/bin/env python3
"""Run and record the audited upstream Alpamayo M0 reference inference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
UPSTREAM_ROOT = ROOT / "third_party" / "alpamayo1.5"
UPSTREAM_SRC = UPSTREAM_ROOT / "src"
for source_root in (SRC, UPSTREAM_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from acarla.model.m0_reference import (  # noqa: E402
    json_safe,
    max_repeat_delta_meters,
    min_ade_meters,
    validate_model_outputs,
    validate_upstream_sample,
)
from acarla.model.readiness import assess_runtime, discover_runtime  # noqa: E402

DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")


def _cuda_memory(torch) -> list[dict]:
    memory = []
    for index in range(torch.cuda.device_count()):
        memory.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
            }
        )
    return memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="existing filesystem location for Hugging Face downloads",
    )
    parser.add_argument("--output", type=Path, default=Path("results/m0_reference_attempt.json"))
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    result = {
        "status": "started",
        "strategy": "upstream_bf16_single_gpu_sdpa",
        "model_id": args.model_id,
        "clip_id": args.clip_id,
        "t0_us": args.t0_us,
        "seed": args.seed,
        "repeats": args.repeats,
        "max_generation_length": args.max_generation_length,
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "project_git_revision": _git_revision(ROOT),
        "upstream_git_revision": _git_revision(UPSTREAM_ROOT),
        "environment": {
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "cuda_visible_devices_set": "CUDA_VISIBLE_DEVICES" in os.environ,
        },
    }

    try:
        readiness = assess_runtime(discover_runtime(args.cache_dir), attention_backend="sdpa")
        result["preflight"] = readiness.to_json_dict()
        if not readiness.reference_ready:
            result["status"] = "blocked_by_preflight"
            _write_result(args.output, result)
            print(f"M0 blocked by preflight; details written to {args.output}")
            return 2

        os.environ["HF_HOME"] = str(args.cache_dir.expanduser().resolve())
        import torch
        from alpamayo1_5 import helper
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)

        data_start = time.perf_counter()
        data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
        result["input"] = validate_upstream_sample(data)
        result["data_load_seconds"] = time.perf_counter() - data_start

        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        load_start = time.perf_counter()
        model = Alpamayo1_5.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        processor = helper.get_processor(model.tokenizer)
        torch.cuda.synchronize()
        result["model_load_seconds"] = time.perf_counter() - load_start

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = helper.to_device(
            {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
            "cuda",
        )

        runs = []
        predictions = []
        for repeat in range(args.repeats):
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = (
                    model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=0.98,
                        temperature=0.6,
                        num_traj_samples=1,
                        max_generation_length=args.max_generation_length,
                        return_extra=True,
                    )
                )
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_start
            xyz, rot = validate_model_outputs(pred_xyz, pred_rot)
            predictions.append(xyz)
            runs.append(
                {
                    "repeat": repeat,
                    "inference_seconds": inference_seconds,
                    "pred_xyz_shape": list(xyz.shape),
                    "pred_xyz_dtype": str(xyz.dtype),
                    "pred_rot_shape": list(rot.shape),
                    "pred_rot_dtype": str(rot.dtype),
                    "min_ade_meters": min_ade_meters(xyz, data["ego_future_xyz"]),
                    "reasoning": extra,
                    "pred_xyz": xyz,
                    "pred_rot": rot,
                }
            )

        result["runs"] = runs
        result["max_repeat_delta_meters"] = max_repeat_delta_meters(predictions)
        result["cuda_memory"] = _cuda_memory(torch)
        result["status"] = "ok"
        _write_result(args.output, result)
        print(f"M0 reference inference completed; result written to {args.output}")
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_result(args.output, result)
        print(f"M0 reference inference failed; details written to {args.output}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
