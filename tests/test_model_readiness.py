from __future__ import annotations

from acarla.model.readiness import (
    GpuInfo,
    RuntimeInfo,
    _base_version,
    assess_runtime,
)


def _runtime(**overrides):
    values = {
        "python_version": "3.12.8",
        "torch_version": "2.8.0+cu128",
        "transformers_version": "4.57.1",
        "operating_system": "Linux",
        "cuda_available": True,
        "gpus": (GpuInfo(0, "reference GPU", 80.0, (9, 0)),),
        "free_space_gib": 100.0,
        "hf_token_present": True,
    }
    values.update(overrides)
    return RuntimeInfo(**values)


def _checks_by_name(report):
    return {check.name: check for check in report.checks}


def _strategies_by_name(report):
    return {strategy.name: strategy for strategy in report.strategies}


def test_reference_machine_passes():
    report = assess_runtime(_runtime())

    assert report.reference_ready
    assert all(check.status != "fail" for check in report.checks)
    assert _strategies_by_name(report)["upstream_bf16_single_gpu"].status == "reference-ready"


def test_two_t4s_are_not_mistaken_for_one_24_gib_gpu():
    t4s = (
        GpuInfo(0, "Tesla T4", 16.0, (7, 5)),
        GpuInfo(1, "Tesla T4", 16.0, (7, 5)),
    )
    report = assess_runtime(_runtime(gpus=t4s))
    checks = _checks_by_name(report)
    strategies = _strategies_by_name(report)

    assert not report.reference_ready
    assert checks["single_gpu_memory"].status == "fail"
    assert checks["native_bf16"].status == "fail"
    assert strategies["fp16_device_map_auto"].status == "eligible-unverified"


def test_six_gib_turing_machine_rejects_all_model_paths():
    local_gpu = (GpuInfo(0, "GTX 1660 Ti", 6.0, (7, 5)),)
    report = assess_runtime(
        _runtime(
            python_version="3.13.5",
            torch_version="2.6.0+cu124",
            transformers_version=None,
            operating_system="Windows",
            gpus=local_gpu,
            free_space_gib=5.2,
            hf_token_present=False,
        )
    )
    checks = _checks_by_name(report)
    strategies = _strategies_by_name(report)

    assert not report.reference_ready
    assert checks["python"].status == "fail"
    assert checks["torch"].status == "fail"
    assert checks["transformers"].status == "fail"
    assert checks["artifact_space"].status == "fail"
    assert checks["hf_token"].status == "fail"
    assert checks["operating_system"].status == "warn"
    assert strategies["fp16_device_map_auto"].status == "ineligible"
    assert strategies["nf4_backbone_fp16_expert"].status == "ineligible"


def test_flash_attention_default_is_not_accepted_for_portable_path():
    report = assess_runtime(_runtime(), attention_backend="flash_attention_2")

    assert not report.reference_ready
    assert _checks_by_name(report)["attention_backend"].status == "fail"


def test_version_parser_ignores_local_cuda_suffix():
    assert _base_version("2.8.0+cu128") == (2, 8, 0)
    assert _base_version("4.57.1") == (4, 57, 1)
    assert _base_version("unknown") is None
