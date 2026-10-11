"""
Helper module for saving PEFT models from checkpoints.
Import these functions from main.py.
"""

import os
import re
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==== Config ====
MODEL_NAME = "bigscience/bloomz-1b1"
OUTPUT_DIR = "./data/prompt_data"
# ================


def list_checkpoints(output_dir):
    """Return a sorted list of (step, path) tuples."""
    ckpts = glob.glob(f"{output_dir}/checkpoint-*")
    result = []
    for c in ckpts:
        m = re.search(r"checkpoint-(\d+)", c)
        if m:
            result.append((int(m.group(1)), c))
    return sorted(result)


def pick_best_checkpoint(checkpoints):
    """
    Pick the best checkpoint based on trainer_state.json's best_model_checkpoint,
    falling back to the latest one.
    """
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found!")

    for _, ckpt_path in reversed(checkpoints):
        state_path = os.path.join(ckpt_path, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            best = state.get("best_model_checkpoint")
            if best and os.path.exists(best):
                print(f"Found best checkpoint in trainer_state: {best}")
                return best

    return checkpoints[-1][1]


def save_peft_from_checkpoint(
    checkpoint_path,
    save_path,
    model_name=MODEL_NAME,
    base_model=None,
    tokenizer=None,
):
    """
    Load a PEFT adapter from a checkpoint and save it to a target path.

    If base_model and tokenizer are provided, they're reused (faster).
    Otherwise they're loaded fresh.
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Load base model if not provided
    if base_model is None:
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load PEFT adapter
    print("Loading PEFT adapter...")
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    peft_model.eval()

    # Save
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving PEFT model to: {save_path}")
    peft_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Show what was saved
    print("\nSaved files:")
    for f in os.listdir(save_path):
        p = os.path.join(save_path, f)
        size = os.path.getsize(p)
        if size > 1024 * 1024:
            print(f"  - {f} ({size / (1024*1024):.2f} MB)")
        else:
            print(f"  - {f} ({size / 1024:.1f} KB)")

    return peft_model, tokenizer
