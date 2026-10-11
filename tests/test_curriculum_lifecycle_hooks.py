"""CPU control-flow regressions; execute production methods with fake actors."""

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


@pytest.fixture
def hooks():
    spec = importlib.util.spec_from_file_location("lifecycle_under_test", ROOT / "slime/utils/lifecycle_hooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_hook_dispatch_and_error_propagation(hooks, monkeypatch):
    assert hooks.call_hook(SimpleNamespace(), "missing", 1) is None
    seen = []
    fake = ModuleType("slime.utils.misc")
    fake.load_function = lambda path: lambda args, value: seen.append((path, args, value)) or "result"
    monkeypatch.setitem(sys.modules, "slime.utils.misc", fake)
    args = SimpleNamespace(callback="my_recipe.hook")
    assert hooks.call_hook(args, "callback", 3) == "result"
    assert seen == [("my_recipe.hook", args, 3)]

    def reject(*_):
        raise ValueError("bad receipt")

    fake.load_function = lambda _: reject
    with pytest.raises(ValueError, match="bad receipt"):
        hooks.call_hook(args, "callback", 3)


def test_manager_save_load_and_completion_order(hooks, monkeypatch):
    fake = ModuleType("slime.data.checkpoint")
    fake.SourceRestore = type("SourceRestore", (), {})
    monkeypatch.setitem(sys.modules, "slime.data.checkpoint", fake)
    events = []

    class Store:
        def save(self, index):
            events.append(("save", self.name, index))

        def load(self, index, **kwargs):
            events.append(("load", self.name, index))

        def training_completed(self, index):
            events.append(("complete", self.name, index))

    def call(args, name, index):
        events.append(("hook", name, index))
        return {"tier_before": 0, "tier_after": 1}

    ns = functions("slime/ray/rollout.py", {"save", "load", "training_completed"}, {"call_hook": call})
    data, batch = Store(), Store()
    data.name, batch.name = "data", "batch"
    manager = SimpleNamespace(args=SimpleNamespace(), data_source=data, batch_builder=batch)
    ns["save"](manager, 4)
    assert events == [
        ("save", "data", 4),
        ("save", "batch", 4),
        ("hook", "custom_rollout_checkpoint_saved_hook_path", 4),
    ]
    events.clear()
    ns["load"](manager, 4)
    assert events == [
        ("hook", "custom_rollout_checkpoint_load_hook_path", 4),
        ("load", "data", 4),
        ("load", "batch", 4),
    ]
    events.clear()
    event = ns["training_completed"](manager, 4)
    assert events[0] == ("complete", "batch", 4)
    assert hooks.completion_requests_eval(event)


def test_bad_restore_receipt_prevents_datasource_load():
    def reject(*_):
        raise ValueError("bad receipt")

    ns = functions("slime/ray/rollout.py", {"load"}, {"call_hook": reject})
    with pytest.raises(ValueError, match="bad receipt"):
        ns["load"](SimpleNamespace(args=None), 4)


@pytest.mark.parametrize("role", ["actor", "ref"])
def test_megatron_callback_observes_restored_scheduler_and_role(role):
    tree = ast.parse((ROOT / "slime/backends/megatron_utils/model.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_model_and_optimizer")
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    events = []
    scheduler = SimpleNamespace(num_steps=0)

    def load(*args, **kwargs):
        scheduler.num_steps = 1920
        events.append("restore")
        return 10, None

    callback = ModuleType("slime.utils.lifecycle_hooks")

    def hook(args, name, loaded_scheduler, iteration, loaded_role):
        assert loaded_scheduler is scheduler and scheduler.num_steps == 1920
        assert iteration == 10 and loaded_role == role
        events.append("hook")

    callback.call_hook = hook
    ns = dict(
        torch=SimpleNamespace(version=SimpleNamespace(hip=False)),
        setup_model_and_optimizer=lambda *args: ([SimpleNamespace()], None, scheduler),
        _critic_output_layer_needs_reinit=lambda *args: False,
        clear_memory=lambda: None,
        load_checkpoint=load,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "model.py", "exec"), ns)
    previous = sys.modules.get("slime.utils.lifecycle_hooks")
    sys.modules["slime.utils.lifecycle_hooks"] = callback
    try:
        ns["initialize_model_and_optimizer"](SimpleNamespace(), role)
    finally:
        if previous is None:
            sys.modules.pop("slime.utils.lifecycle_hooks")
        else:
            sys.modules["slime.utils.lifecycle_hooks"] = previous
    assert events == ["restore", "hook"]


@pytest.mark.parametrize(
    "event,expected",
    [
        (None, False),
        ({}, False),
        ({"tier_before": 1}, False),
        ({"tier_before": 1, "tier_after": 1}, False),
        ({"tier_before": 1, "tier_after": 2}, True),
        ({"evaluate": True}, True),
    ],
)
def test_eval_request(hooks, event, expected):
    assert hooks.completion_requests_eval(event) is expected


def test_durable_hook_requires_synchronous_checkpoint(hooks):
    hooks.validate_lifecycle_args(SimpleNamespace())
    args = SimpleNamespace(custom_rollout_checkpoint_saved_hook_path="recipe.saved", force_sync_save=False)
    with pytest.raises(ValueError, match="force-sync-save"):
        hooks.validate_lifecycle_args(args)
    args.force_sync_save = True
    hooks.validate_lifecycle_args(args)


@pytest.mark.parametrize("start,opt_in,expected_initial", [(0, False, True), (10, False, False), (10, True, True)])
@pytest.mark.parametrize("force_sync,changed", [(True, True), (False, False)])
def test_train_resume_eval_sync_save_and_tier_event(hooks, start, opt_in, expected_initial, force_sync, changed, monkeypatch):
    events = []

    class Remote:
        def __init__(self, owner, name):
            self.owner, self.name = owner, name

        def remote(self, *a, **k):
            return self.owner.invoke(self.name, *a, **k)

    class Actor:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, name):
            return Remote(self, name)

        def invoke(self, name, *a, **k):
            events.append((self.name, name, a, k))
            if name == "training_completed":
                return {"tier_before": 0, "tier_after": 1} if changed else None

    class TrainActor:
        def __getattr__(self, name):
            def call(*a, **k):
                events.append(("actor", name, a, k))

            return call

    args = SimpleNamespace(
        release_train=False,
        rollout_data_transport="ray", debug_train_only=False, debug_rollout_only=False,
        offload_rollout=False,
        offload_train=True,
        check_weight_update_equal=False,
        num_rollout=start + 2,
        eval_interval=5,
        start_rollout_id=start,
        skip_eval_before_train=False,
        use_critic=False,
        save_interval=1,
        force_sync_save=force_sync,
        eval_before_train_on_resume=opt_in,
    )
    ns = dict(
        ray=SimpleNamespace(get=lambda x: x),
        configure_logger=lambda: None,
        init_tracking=lambda _: None,
        finish_tracking=lambda _: None,
        create_placement_groups=lambda _: {"rollout": None},
        create_rollout_manager=lambda *_, **kwargs: (Actor("manager"), 1),
        create_training_models=lambda *_: (TrainActor(), None),
        should_run_periodic_action=lambda i, interval, *a: interval == 1,
        completion_requests_eval=hooks.completion_requests_eval,
        validate_lifecycle_args=hooks.validate_lifecycle_args,
    )
    monkeypatch.setitem(sys.modules, "ray", ns["ray"])
    functions("slime/data/checkpoint.py", {"save_checkpoint"}, ns)
    functions("train.py", {"train"}, ns)["train"](args)
    eval_events = [e for e in events if e[1] == "eval"]
    assert len(eval_events) == 2 * int(changed) + int(expected_initial)
    saves = [e for e in events if e[1] == "save_model"]
    assert saves[0][3]["force_sync"] is force_sync
    assert saves[-1][3]["force_sync"] is True
    for index in (start, start + 1):
        complete = next(i for i, e in enumerate(events) if e[1] == "training_completed" and e[2] == (index,))
        save = next(i for i, e in enumerate(events) if e[1] == "save_model" and e[2] == (index,))
        rollout_save = next(i for i, e in enumerate(events) if e[1] == "save" and e[2] == (index,))
        assert complete < save < rollout_save


def test_cli_hook_defaults_and_opt_in():
    names = {"add_rollout_arguments", "add_eval_arguments", "add_custom_megatron_plugins_arguments"}
    ns = functions("slime/utils/arguments.py", names | {"reset_arg"}, {})
    parser = argparse.ArgumentParser()
    for name in names:
        ns[name](parser)
    defaults = parser.parse_args([])
    assert defaults.custom_rollout_checkpoint_load_hook_path is None
    assert defaults.custom_megatron_after_checkpoint_load_hook_path is None
    assert not defaults.force_sync_save and not defaults.eval_before_train_on_resume
    args = parser.parse_args(
        [
            "--force-sync-save",
            "--eval-before-train-on-resume",
            "--custom-rollout-checkpoint-load-hook-path",
            "recipe.restore",
        ]
    )
    assert args.force_sync_save and args.eval_before_train_on_resume
    assert args.custom_rollout_checkpoint_load_hook_path == "recipe.restore"
