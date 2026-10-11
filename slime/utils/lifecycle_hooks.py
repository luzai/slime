"""Optional synchronous hooks for durable rollout and optimizer state."""

from collections.abc import Mapping


def call_hook(args, attribute, *values):
    path = getattr(args, attribute, None)
    if path is None:
        return None
    from slime.utils.misc import load_function

    return load_function(path)(args, *values)


def completion_requests_eval(event):
    """Accept an explicit request or a curriculum tier transition."""
    if not isinstance(event, Mapping):
        return False
    if event.get("evaluate"):
        return True
    return "tier_before" in event and "tier_after" in event and event["tier_before"] != event["tier_after"]


def validate_lifecycle_args(args):
    if getattr(args, "custom_rollout_checkpoint_saved_hook_path", None) and not getattr(
        args, "force_sync_save", False
    ):
        raise ValueError("a rollout checkpoint-saved hook requires --force-sync-save")
