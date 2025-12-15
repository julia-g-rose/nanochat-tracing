import os
import time

import wandb

try:
    import weave
except Exception:  # weave isn't an install requirement in all environments
    weave = None


def init_weave(
    wandb_run=None,
    project_env: str = "WANDB_PROJECT",
    default_project: str = "nanochat",
    entity_env: str = "WANDB_ENTITY",
    retries: int = 10,
    retry_delay: float = 0.1,
    warn_fn=print,
):
    """
    Initialize Weave tracing using W&B project/entity.

    - Prefers explicit env vars (WANDB_ENTITY, WANDB_PROJECT).
    - Falls back to wandb_run.entity (with brief retries) if available.
    - Falls back to wandb.Api().default_entity last.
    - Returns True on success, False otherwise.
    """
    project = os.environ.get(project_env, default_project)
    entity = os.environ.get(entity_env)

    if not entity and wandb_run is not None:
        for _ in range(retries):
            entity = getattr(wandb_run, "entity", None)
            if entity:
                break
            time.sleep(retry_delay)

    if not entity:
        try:
            entity = wandb.Api().default_entity
        except Exception:
            entity = None

    if entity and project:
        try:
            if weave is None:
                if warn_fn:
                    warn_fn("⚠️ Weave is not installed; skipping weave.init")
                return False
            weave.init(f"{entity}/{project}")
            if warn_fn:
                warn_fn(f"✅ Weave tracing initialized: {entity}/{project}")
            return True
        except Exception as e:
            if warn_fn:
                warn_fn(f"⚠️ Could not initialize Weave tracing: {e}")
            return False

    if warn_fn:
        warn_fn("⚠️ Could not initialize Weave tracing: wandb entity/project not available")
        warn_fn("   💡 Set WANDB_ENTITY (and optionally WANDB_PROJECT) to enable Weave tracing")
    return False

