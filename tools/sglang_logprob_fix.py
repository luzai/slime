"""Pinned, idempotent SGLang consumer fix; run before importing the engine."""
import argparse
import hashlib
import os
from pathlib import Path

ORIGINAL = "4c6dd7cb413155d61be5193f1dd90ccf844b8d7c0daedfaa3973dc9e30b85e03"
FIXED = "9fc6471b0cab34fe53c412888a0cc02887bb62eaae78a046be38af48f5604d66"
DEFAULT = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py"


def transform(raw):
    digest = hashlib.sha256(raw).hexdigest()
    if digest == FIXED:
        return raw
    if digest != ORIGINAL:
        raise RuntimeError(f"Unreviewed SGLang source {digest}; re-audit before training")
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for item, field in (("v", "next_token_top_logprobs_val"),
                        ("x", "next_token_top_logprobs_idx"),
                        ("v", "next_token_token_ids_logprobs_val")):
        old = f"{item}.tolist() for {item} in logits_output.{field}"
        new = f"{item}.tolist() if torch.is_tensor({item}) else {item} for {item} in logits_output.{field}"
        if text.count(old) != 2:
            raise RuntimeError(f"Unexpected consumer sites: {field}")
        text = text.replace(old, new)
    result = text.encode("utf-8")
    compile(text, DEFAULT, "exec")
    if hashlib.sha256(result).hexdigest() != FIXED:
        raise RuntimeError("Patched source identity mismatch")
    return result


def apply(path):
    # Serialize concurrent launchers in the same Linux container.
    import fcntl
    with path.with_suffix(path.suffix + ".slime-fix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = path.read_bytes()
        fixed = transform(raw)
        if raw != fixed:
            temp = path.with_name(path.name + f".slime-fix-{os.getpid()}")
            try:
                temp.write_bytes(fixed)
                temp.chmod(path.stat().st_mode)
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
    print(f"SGLANG_LOGPROB_FIX_VERIFIED sha256={FIXED}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(DEFAULT))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    apply(args.source)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if command:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
