"""Conservative M0 readiness checks for an Alpamayo 1.5 worker.

The checks in this module deliberately distinguish the upstream reference
path from experimental memory-saving strategies.  Passing an eligibility
check for NF4 or multi-GPU placement is not evidence that the model runs; only
an actual inference can promote such a strategy to "verified".
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

REQUIRED_PYTHON = (3, 12)
REQUIRED_TORCH = (2, 8, 0)
REQUIRED_TRANSFORMERS = (4, 57, 1)
MIN_SINGLE_GPU_MEMORY_GIB = 24.0
MIN_MODEL_ARTIFACT_SPACE_GIB = 22.0

Status = Literal["pass", "fail", "warn"]


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_gib: float
    compute_capability: tuple[int, int]

    @property
    def native_bf16(self) -> bool:
        return self.compute_capability[0] >= 8


@dataclass(frozen=True)
class RuntimeInfo:
    python_version: str
    torch_version: str | None
    transformers_version: str | None
    operating_system: str
    cuda_available: bool
    gpus: tuple[GpuInfo, ...]
    free_space_gib: float
    hf_token_present: bool


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class StrategyAssessment:
    name: str
    status: Literal["reference-ready", "eligible-unverified", "ineligible"]
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    runtime: RuntimeInfo
    checks: tuple[Check, ...]
    strategies: tuple[StrategyAssessment, ...]

    @property
    def reference_ready(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reference_ready"] = self.reference_ready
        return result


def _base_version(version: str) -> tuple[int, int, int] | None:
    """Return the first semantic-version triple, ignoring CUDA/local suffixes."""
    parts = version.split("+", 1)[0].split(".")
    if len(parts) < 3:
        return None
    numbers: list[int] = []
    for part in parts[:3]:
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            return None
        numbers.append(int(digits))
    return tuple(numbers)  # type: ignore[return-value]


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def discover_runtime(cache_dir: str | Path = ".") -> RuntimeInfo:
    """Inspect the current process without loading model weights or printing secrets."""
    torch_version = _package_version("torch")
    transformers_version = _package_version("transformers")
    cuda_available = False
    gpus: list[GpuInfo] = []

    if torch_version is not None:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    gpus.append(
                        GpuInfo(
                            index=index,
                            name=props.name,
                            memory_gib=props.total_memory / 1024**3,
                            compute_capability=(props.major, props.minor),
                        )
                    )
        except (ImportError, RuntimeError):
            cuda_available = False
            gpus = []

    cache_path = Path(cache_dir).expanduser().resolve()
    free_space_gib = shutil.disk_usage(cache_path).free / 1024**3
    token_present = bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _nonempty_file(Path.home() / ".cache" / "huggingface" / "token")
    )
    return RuntimeInfo(
        python_version=platform.python_version(),
        torch_version=torch_version,
        transformers_version=transformers_version,
        operating_system=platform.system(),
        cuda_available=cuda_available,
        gpus=tuple(gpus),
        free_space_gib=free_space_gib,
        hf_token_present=token_present,
    )


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def assess_runtime(runtime: RuntimeInfo, attention_backend: str = "sdpa") -> ReadinessReport:
    """Evaluate the audited upstream reference path and experimental candidates."""
    checks: list[Check] = []

    python_ok = _major_minor(runtime.python_version) == REQUIRED_PYTHON
    checks.append(
        Check(
            "python",
            "pass" if python_ok else "fail",
            f"found {runtime.python_version}; Alpamayo pins Python 3.12.*",
        )
    )
    checks.append(
        _exact_package_check("torch", runtime.torch_version, REQUIRED_TORCH)
    )
    checks.append(
        _exact_package_check(
            "transformers", runtime.transformers_version, REQUIRED_TRANSFORMERS
        )
    )
    checks.append(
        Check(
            "cuda",
            "pass" if runtime.cuda_available and runtime.gpus else "fail",
            f"detected {len(runtime.gpus)} CUDA GPU(s)",
        )
    )

    largest_gpu = max((gpu.memory_gib for gpu in runtime.gpus), default=0.0)
    checks.append(
        Check(
            "single_gpu_memory",
            "pass" if largest_gpu >= MIN_SINGLE_GPU_MEMORY_GIB else "fail",
            f"largest GPU {largest_gpu:.1f} GiB; reference path requires at least "
            f"{MIN_SINGLE_GPU_MEMORY_GIB:.0f} GiB on one GPU",
        )
    )
    bf16_gpus = [gpu for gpu in runtime.gpus if gpu.native_bf16]
    checks.append(
        Check(
            "native_bf16",
            "pass" if bf16_gpus else "fail",
            "at least one sm_80+ GPU detected"
            if bf16_gpus
            else "no sm_80+ GPU detected; FP16 is an unverified deviation",
        )
    )
    checks.append(
        Check(
            "attention_backend",
            "pass" if attention_backend == "sdpa" else "fail",
            f"configured {attention_backend!r}; use 'sdpa' to avoid the FlashAttention-2 default",
        )
    )
    checks.append(
        Check(
            "artifact_space",
            "pass" if runtime.free_space_gib >= MIN_MODEL_ARTIFACT_SPACE_GIB else "fail",
            f"{runtime.free_space_gib:.1f} GiB free; model download is approximately "
            f"{MIN_MODEL_ARTIFACT_SPACE_GIB:.0f} GiB",
        )
    )
    checks.append(
        Check(
            "hf_token",
            "pass" if runtime.hf_token_present else "fail",
            "Hugging Face credential is present"
            if runtime.hf_token_present
            else "no credential found; gated weights cannot be downloaded",
        )
    )
    checks.append(
        Check(
            "operating_system",
            "pass" if runtime.operating_system == "Linux" else "warn",
            f"found {runtime.operating_system}; upstream examples are validated on Linux",
        )
    )

    aggregate_memory = sum(gpu.memory_gib for gpu in runtime.gpus)
    multi_gpu_eligible = len(runtime.gpus) >= 2 and aggregate_memory >= MIN_SINGLE_GPU_MEMORY_GIB
    nf4_eligible = bool(runtime.gpus) and largest_gpu >= 8.0
    strategies = (
        StrategyAssessment(
            "upstream_bf16_single_gpu",
            "reference-ready" if all(check.status != "fail" for check in checks) else "ineligible",
            "Audited upstream path; the only strategy this preflight may mark verified-ready.",
        ),
        StrategyAssessment(
            "fp16_device_map_auto",
            "eligible-unverified" if multi_gpu_eligible else "ineligible",
            f"{len(runtime.gpus)} GPU(s), {aggregate_memory:.1f} GiB aggregate. Aggregate VRAM "
            "does not prove that Hugging Face can place the custom model correctly.",
        ),
        StrategyAssessment(
            "nf4_backbone_fp16_expert",
            "eligible-unverified" if nf4_eligible else "ineligible",
            f"largest GPU {largest_gpu:.1f} GiB. Backbone-only quantization is a project "
            "proposal, not an upstream-supported or locally verified path.",
        ),
    )
    return ReadinessReport(runtime=runtime, checks=tuple(checks), strategies=strategies)


def _major_minor(version: str) -> tuple[int, int] | None:
    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return int(parts[0]), int(parts[1])


def _exact_package_check(
    name: str, found: str | None, required: tuple[int, int, int]
) -> Check:
    required_text = ".".join(str(part) for part in required)
    if found is None:
        return Check(name, "fail", f"not installed; Alpamayo pins {name}=={required_text}")
    ok = _base_version(found) == required
    return Check(
        name,
        "pass" if ok else "fail",
        f"found {found}; Alpamayo pins {name}=={required_text}",
    )


def format_report(report: ReadinessReport) -> str:
    lines = [
        "Alpamayo 1.5 M0 readiness",
        f"Reference path ready: {'YES' if report.reference_ready else 'NO'}",
        "",
        "Checks:",
    ]
    for check in report.checks:
        lines.append(f"  [{check.status.upper():4s}] {check.name}: {check.detail}")
    lines.append("")
    lines.append("Strategies:")
    for strategy in report.strategies:
        lines.append(f"  [{strategy.status}] {strategy.name}: {strategy.detail}")
    return "\n".join(lines)
