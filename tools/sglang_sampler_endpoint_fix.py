"""Pinned fix for the seeded sampler's inclusive upper uniform endpoint."""

import argparse
import hashlib
import os
from pathlib import Path

ORIGINAL = "c2a53326caa12f9651d2d8ce6e921678c72cc169c2914750a36fdbaa33ca4238"
FIXED = "ddc52a5fface4901db4492248cdd1e6c9697a295e223188024e22d7a864947ef"
DEFAULT = "/sgl-workspace/sglang/python/sglang/srt/layers/sampler.py"


def transform(raw):
    digest = hashlib.sha256(raw).hexdigest()
    if digest == FIXED:
        return raw
    if digest != ORIGINAL:
        raise RuntimeError(f"Unreviewed SGLang sampler {digest}; re-audit before training")
    text = raw.decode("utf-8")
    old = "    x = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max\n"
    if text.count(old) != 1:
        raise RuntimeError("Unexpected seeded sampler conversion sites")
    text = text.replace(old, old + "    x.clamp_(max=1.0 - torch.finfo(torch.float64).eps)\n")
    result = text.encode("utf-8")
    compile(text, DEFAULT, "exec")
    if hashlib.sha256(result).hexdigest() != FIXED:
        raise RuntimeError("Patched sampler identity mismatch")
    return result


def apply(path):
    import fcntl

    with path.with_suffix(path.suffix + ".slime-fix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = path.read_bytes()
        fixed = transform(raw)
        if raw != fixed:
            temporary = path.with_name(path.name + f".slime-fix-{os.getpid()}")
            try:
                temporary.write_bytes(fixed)
                temporary.chmod(path.stat().st_mode)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    print(f"SGLANG_SAMPLER_ENDPOINT_FIX_VERIFIED sha256={FIXED}", flush=True)


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
