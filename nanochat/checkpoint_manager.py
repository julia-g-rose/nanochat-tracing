"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0, wandb_run=None):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
        
        # Upload checkpoint to wandb as artifact if wandb_run is provided
        if wandb_run is not None and hasattr(wandb_run, 'log_artifact'):
            try:
                import wandb
                # Create artifact with metadata
                artifact_name = f"checkpoint-step-{step}"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    metadata={
                        "step": step,
                        **meta_data
                    }
                )
                
                # Add checkpoint files
                artifact.add_file(model_path, name=f"model_{step:06d}.pt")
                artifact.add_file(meta_path, name=f"meta_{step:06d}.json")
                
                # Add tokenizer files (critical for inference!)
                base_dir = get_base_dir()
                tokenizer_dir = os.path.join(base_dir, "tokenizer")
                tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
                token_bytes_pt = os.path.join(tokenizer_dir, "token_bytes.pt")
                
                if os.path.exists(tokenizer_pkl):
                    artifact.add_file(tokenizer_pkl, name="tokenizer/tokenizer.pkl")
                    logger.info(f"Added tokenizer to artifact: {tokenizer_pkl}")
                else:
                    logger.warning(f"Tokenizer not found at {tokenizer_pkl}, skipping")
                
                if os.path.exists(token_bytes_pt):
                    artifact.add_file(token_bytes_pt, name="tokenizer/token_bytes.pt")
                    logger.info(f"Added token_bytes to artifact: {token_bytes_pt}")
                else:
                    logger.warning(f"Token bytes not found at {token_bytes_pt}, skipping")
                
                # Log artifact with step as alias
                wandb_run.log_artifact(artifact, aliases=[f"step_{step}", "latest"])
                logger.info(f"✅ Uploaded checkpoint to wandb as artifact: {artifact_name}")
                
            except Exception as e:
                logger.warning(f"Failed to upload checkpoint to wandb: {e}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def download_checkpoint_from_wandb(entity, project, artifact_name, download_dir=None):
    """
    Download a checkpoint artifact from WandB.
    
    Args:
        entity: WandB entity (username or team name)
        project: WandB project name
        artifact_name: Name or alias of the artifact (e.g., "checkpoint-step-1000:latest")
        download_dir: Optional directory to download to. If None, uses a temp directory.
    
    Returns:
        Path to the downloaded artifact directory
    """
    try:
        import wandb
        api = wandb.Api()
        artifact_path = f"{entity}/{project}/{artifact_name}"
        logger.info(f"Downloading artifact from WandB: {artifact_path}")
        artifact = api.artifact(artifact_path)
        download_path = artifact.download(root=download_dir)
        logger.info(f"✅ Downloaded artifact to: {download_path}")
        
        # If tokenizer files were included, copy them to the expected location
        tokenizer_pkl = os.path.join(download_path, "tokenizer", "tokenizer.pkl")
        token_bytes_pt = os.path.join(download_path, "tokenizer", "token_bytes.pt")
        
        if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_pt):
            base_dir = get_base_dir()
            tokenizer_dir = os.path.join(base_dir, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            
            import shutil
            shutil.copy2(tokenizer_pkl, os.path.join(tokenizer_dir, "tokenizer.pkl"))
            shutil.copy2(token_bytes_pt, os.path.join(tokenizer_dir, "token_bytes.pt"))
            logger.info(f"✅ Copied tokenizer files to: {tokenizer_dir}")
        
        return download_path
    except Exception as e:
        logger.error(f"Failed to download checkpoint from wandb: {e}")
        raise

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model_from_wandb(entity, project, artifact_name, device, phase):
    """
    Load a model directly from a WandB artifact.
    
    Args:
        entity: WandB entity (username or team name)
        project: WandB project name
        artifact_name: Name or alias of the artifact (e.g., "checkpoint-step-1000:latest")
        device: Device to load the model on
        phase: "train" or "eval"
    
    Returns:
        model, tokenizer, meta_data
    """
    # Download the artifact
    artifact_dir = download_checkpoint_from_wandb(entity, project, artifact_name)
    
    # Find the model file in the downloaded artifact
    model_files = glob.glob(os.path.join(artifact_dir, "model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model file found in artifact: {artifact_dir}")
    
    # Extract step from filename
    model_file = model_files[0]
    step = int(os.path.basename(model_file).split("_")[-1].split(".")[0])
    
    # Build and return the model
    return build_model(artifact_dir, step, device, phase)

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
