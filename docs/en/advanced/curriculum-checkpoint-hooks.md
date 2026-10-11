# Optional curriculum checkpoint hooks

These hooks replace the Pacman experiment's direct imports of private control
modules. They are disabled by default and require no Pacman dependency. Recipe
sampling, maps, controller state, seed domains, and budgets remain in the recipe
repository. None of the flags silently resumes or changes a running experiment.

| Flag | Function signature and ordering |
| --- | --- |
| `--custom-rollout-training-completed-hook-path` | `(args, rollout_id)`, after successful optimizer work and batch-builder completion. Return `{"evaluate": true}` or differing `tier_before`/`tier_after` values to request evaluation after weight synchronization. |
| `--custom-rollout-checkpoint-saved-hook-path` | `(args, rollout_id)`, after synchronous model save and datasource/batch-builder save. Publish the joint receipt here. Requires `--force-sync-save`. |
| `--custom-rollout-checkpoint-load-hook-path` | `(args, rollout_id)`, before datasource/batch-builder restore. Reject mismatched or incomplete receipts here. |
| `--custom-megatron-after-checkpoint-load-hook-path` | `(args, scheduler, iteration, role)`, after Megatron model/optimizer/RNG restore on each rank. Actor-only schedule changes must explicitly check `role == "actor"`. |

Hook exceptions propagate and stop the run. Hooks must be synchronous, available
in the worker environment, and idempotent where initialization can repeat.
The framework does not create joint receipts, verify a budget, or reset RNG;
those responsibilities belong to the recipe hooks and launch driver.

`--force-sync-save` waits for every **scheduled** model save before rollout state
is saved; it does not change the checkpoint frequency. A joint-receipt callback
must validate all actor/critic files before publishing completion. The
synchronous training driver (`train.py`) supports these hooks; fully asynchronous
drivers are not covered by this acceptance.

`--eval-before-train-on-resume` evaluates at `start_rollout_id` instead of only at
zero. `--skip-eval-before-train` still takes precedence. This replaces the
experiment's hard-coded `rollout_id == 10`. Without the new flag, initial
evaluation behavior is unchanged.

For the old Pacman control modules, configure the existing
`slime_pacman.formal_rollout.optimizer_completed_hook`,
`checkpoint_saved_hook`, and `checkpoint_load_hook` as the rollout callbacks.
The scheduler callback needs a recipe wrapper with the new four-argument
signature, calling the reviewed scheduler logic for the actor only. Do not copy
the old `index9`, `index10`, or final-sample constants into the framework.
Regenerate source-bound launch manifests after integrating the recipe adapters.

## Pinned serving and game runtime

The audited jobs used `pacman-slime:20260928-d869087e199f` (88 layers), PyTorch
2.11.0+cu129, transformers 5.12.1, SGLang 0.5.15.post1, and pygame 2.6.1.
This records the tested boundary, not compatibility with arbitrary new versions.

Before starting **each fresh worker container**, apply/verify both serving fixes:

```bash
python tools/sglang_sampler_endpoint_fix.py
python tools/sglang_logprob_fix.py
```

Both reject unexpected original hashes and accept their exact fixed hashes.
For read-only native-source mounts, generate and verify fixed files outside
the container, then mount them at the tools' documented native paths. Do not
patch live inference services. The original image remains unchanged.

The sampler fix only clamps the uint32 maximum before the Gumbel transform.
All other hashed values remain unchanged. Without it, uniform value 1 creates
infinite noise and can select an otherwise masked token. The fixed source SHA
is `ddc52a5fface4901db4492248cdd1e6c9697a295e223188024e22d7a864947ef`.
The overlap list/Tensor consumer fix was already published on the base branch;
it is reused here without changing its six-site patch or overlap settings.

## Acceptance boundary

Focused CPU tests cover real training-driver control flow with fake actors,
hook ordering and failures, resume evaluation, synchronized saves, CLI defaults,
dense/MoE classification, empty accelerator variables, and sampler endpoints.
The sampler transform was also checked against the exact remote frozen source
and fixed SHA. These checks are not a new distributed GPU resume acceptance;
the generalized hooks require recipe integration and a dedicated GPU acceptance
before replacing any frozen training source.
