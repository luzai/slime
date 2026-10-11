import ast
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), scope)
    return scope


def test_qwen35_dense_optional_moe_attributes_do_not_classify_it_as_moe():
    ns = extract(
        "slime/backends/megatron_utils/arguments.py",
        {"_is_moe_config", "_has_dense_moe_layers", "_hf_validate_args"},
        {},
    )
    dense = SimpleNamespace(
        model_type="qwen3_5_text",
        moe_intermediate_size=None,
        shared_expert_intermediate_size=None,
        intermediate_size=16,
    )
    args = SimpleNamespace(
        ffn_hidden_size=16, moe_ffn_hidden_size=99, moe_shared_expert_intermediate_size=99, rotary_base=None
    )
    assert not ns["_is_moe_config"](dense)
    ns["_hf_validate_args"](args, dense)
    dense.intermediate_size = 17
    with pytest.raises(AssertionError):
        ns["_hf_validate_args"](args, dense)
    moe = SimpleNamespace(model_type="qwen3_5_moe", num_experts=16)
    assert ns["_is_moe_config"](moe)


@pytest.mark.parametrize(
    "kind,key",
    [("musa", "MUSA_VISIBLE_DEVICES"), ("npu", "ASCEND_RT_VISIBLE_DEVICES"), ("supa", "SUPA_VISIBLE_DEVICES")],
)
def test_empty_accelerator_environment_does_not_enable_backend(monkeypatch, kind, key):
    names = {f"is_{kind}_environment", f"_{kind}_requested"}
    scope = dict(os=os, **{f"is_{kind}_available": lambda: False})
    ns = extract("slime/utils/accelerator/__init__.py", names, scope)
    for k in [
        "SLIME_ACCELERATOR",
        "MUSA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "SUPA_VISIBLE_DEVICES",
        "MUSA_PATCH_PATH",
        "ASCEND_HOME_PATH",
        "BIREN_HOME",
    ]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(key, "")
    assert not ns[f"_{kind}_requested"]()
    assert not ns[f"is_{kind}_environment"]()
    monkeypatch.setenv(key, "0")
    assert ns[f"_{kind}_requested"]()
    assert ns[f"is_{kind}_environment"]()


def test_sampler_unknown_source_is_rejected_and_endpoints_remain_finite():
    spec = importlib.util.spec_from_file_location("endpoint_fix", ROOT / "tools/sglang_sampler_endpoint_fix.py")
    fix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix)
    with pytest.raises(RuntimeError, match="Unreviewed"):
        fix.transform(b"different sampler")
    maximum = 2**32 - 1
    cap = 1.0 - 2**-52
    for hashed in (0, 1, maximum // 2, maximum - 1, maximum):
        original = hashed / maximum
        clipped = min(original, cap)
        if hashed != maximum:
            assert clipped == original
        log_x = math.log(clipped) if clipped else -float.fromhex("0x1.fffffffffffffp+1023")
        noise = -math.log(-log_x)
        assert math.isfinite(noise)
    # Even maximal Gumbel noise cannot resurrect a masked (-inf) token.
    assert (-math.inf + noise) == -math.inf
