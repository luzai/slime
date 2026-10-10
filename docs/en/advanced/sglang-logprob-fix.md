# Pinned SGLang Tensor/list logprob fix

The frozen SGLang 0.5.15.post1 runtime can return lists and tensors in the
same batch. Six native prefill/decode consumer sites unconditionally call
`.tolist()`. The issue can occur with overlap disabled as well as enabled.
See [upstream issue 34719](https://github.com/sgl-project/sglang/issues/34719).

`tools/sglang_logprob_fix.py` guards exactly those six sites, preserving existing
lists. It checks the original and fixed SHA256, is idempotent, serializes local
launchers, and refuses unreviewed source versions. Sampling math is unchanged.

## New training launches on each server

Pull the reviewed commit into the isolated slime checkout on every worker node.
Inside each fresh locked training container, before any SGLang process starts:

```bash
python /path/to/slime/tools/sglang_logprob_fix.py -- bash /path/to/train.sh
```

The wrapper verifies/applies the fix before executing the existing training
entry point. For Ray multinode jobs, run the verification in **every worker
container** before starting Ray workers; wrapping only the head is insufficient.
Direct external SGLang services also need verification before their launch.

Use the same locked slime image and existing sampler patch. Git push alone does
not update installed SGLang. This wrapper modifies the current container's
installed file, not the image; run it again in every fresh container. A read-only
source mount needs the already-fixed file mounted at the native source path.
Do not apply it to a running service. This change does not automatically change
overlap flags; independently verify `disable_overlap_schedule=False` when desired.

## Validation

H100_1_1 isolated acceptance: 36 CPU cases passed; overlap-enabled candidate
completed 134 requests (62 masked, 24 plain, 24 top-k, 24 pure prefill).
38 fixed inputs matched original overlap-off actions and returned logprobs exactly.
The original overlap-off arm crashed during mixed requests with the same type
error. Pure prefill covered the zero-pruned placeholder window, not positive
length scoring. Long-duration training and throughput benefits remain unverified.

Independent evidence is in the workspace report
`reports/sglang-overlap-fix-20261009/README.md`; production jobs were not modified.
