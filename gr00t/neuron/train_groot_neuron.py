#!/usr/bin/env python3
"""Standalone mini GR00T N1.6 training script for Neuron.

Trains the minified GR00T N1.6 action head end-to-end (Eagle backbone +
action head) on synthetic data.  Only requires the Isaac-GR00T package
(``gr00t``) and standard PyTorch / HuggingFace libraries.

Architecture (same structure as the full model, just smaller):
  SigLIP2(mini) -> Qwen3(mini) -> AlternateVLDiT(mini) -> Actions

The backbone is frozen -- only the action head is trained, matching the
real GR00T fine-tuning setup.  Backbone features are cached once and
reused across all training steps.

Usage:
    python train_groot_neuron.py [--device cpu|neuron] [--steps 500]
"""

import argparse
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead

# ──────────────────────────────────────────────────────────────────────
# Config paths
# ──────────────────────────────────────────────────────────────────────
_GR00T_PACKAGE_ROOT = os.path.dirname(
    os.path.abspath(os.path.join(__import__("gr00t").__file__))
)
EAGLE_CONFIG_PATH = os.path.join(
    _GR00T_PACKAGE_ROOT, "model", "modules", "nvidia", "Eagle-Block2A-2B-v2"
)
ACTION_HEAD_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gr00t_mini_config"
)

# Input image size (not stored in configs -- a design choice)
IMAGE_H, IMAGE_W = 224, 224


# ──────────────────────────────────────────────────────────────────────
# Config builders
# ──────────────────────────────────────────────────────────────────────

def build_eagle_config():
    """Load the mini Eagle3_VL config from config_mini.json."""
    return AutoConfig.from_pretrained(
        EAGLE_CONFIG_PATH, trust_remote_code=True, _configuration_file="config_mini.json"
    )


def build_gr00t_action_head_config():
    """Load the mini Gr00tN1d6Config from config.json."""
    return Gr00tN1d6Config.from_pretrained(ACTION_HEAD_CONFIG_PATH)


# ──────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────

def build_model(device):
    """Build the complete mini GR00T model from scratch with random weights."""

    print("=" * 60)
    print(f"Building mini GR00T N1.6 model for {device}")
    print("=" * 60)

    eagle_config = build_eagle_config()
    action_config = build_gr00t_action_head_config()

    # --- Build Eagle backbone ---
    print("\n[1/3] Building mini Eagle backbone (SigLIP2 + Qwen3)...")
    eagle_model = AutoModel.from_config(
        eagle_config, trust_remote_code=True, attn_implementation="eager"
    )

    # Trim LLM layers to select_layer (same logic as EagleBackbone.__init__)
    select_layer = action_config.select_layer
    while len(eagle_model.language_model.model.layers) > select_layer:
        eagle_model.language_model.model.layers.pop(-1)

    eagle_model = eagle_model.float().to(device)
    eagle_model.eval()

    n_eagle = sum(p.numel() for p in eagle_model.parameters())
    print(f"    Eagle params: {n_eagle:,} ({n_eagle * 4 / 1024 / 1024:.1f} MB)")

    # --- Build action head ---
    print("\n[2/3] Building mini action head (AlternateVLDiT + MLPs)...")
    action_head = Gr00tN1d6ActionHead(action_config)
    action_head = action_head.float()
    action_head.to(device)
    action_head.eval()

    n_action = sum(p.numel() for p in action_head.parameters())
    print(f"    Action head params: {n_action:,} ({n_action * 4 / 1024 / 1024:.1f} MB)")

    # --- Summary ---
    n_total = n_eagle + n_action
    print("\n[3/3] Model summary:")
    print(f"    Total params: {n_total:,}")
    print(f"    Total size: {n_total * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"    Device: {device}")

    return eagle_model, action_head, eagle_config, action_config


# ──────────────────────────────────────────────────────────────────────
# Dummy input construction
# ──────────────────────────────────────────────────────────────────────

def create_dummy_inputs(eagle_config, action_config, device):
    """Create dummy inputs for both backbone and action head."""
    batch_size = 1

    image_token_index = eagle_config.image_token_index
    patch_size = eagle_config.vision_config.patch_size
    downsample_ratio = eagle_config.downsample_ratio
    num_vision_tokens = (
        int(IMAGE_H // patch_size * downsample_ratio) * int(IMAGE_W // patch_size * downsample_ratio)
    )

    # --- Backbone inputs ---
    pixel_values = [torch.randn(1, 3, IMAGE_H, IMAGE_W).to(device)]

    num_text_tokens = 10
    seq_len = num_text_tokens + num_vision_tokens + 2  # +2 for BOS/EOS
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)
    input_ids[0, 0] = 1  # BOS
    input_ids[0, 1: 1 + num_text_tokens] = torch.randint(100, 1000, (num_text_tokens,)).to(device)
    input_ids[0, 1 + num_text_tokens: 1 + num_text_tokens + num_vision_tokens] = image_token_index
    input_ids[0, 1 + num_text_tokens + num_vision_tokens] = 2  # EOS

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

    backbone_inputs = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # --- Action head inputs ---
    state = torch.randn(batch_size, 1, action_config.max_state_dim).to(device)
    action = torch.randn(batch_size, action_config.action_horizon, action_config.max_action_dim).to(device)
    embodiment_id = torch.zeros(batch_size, dtype=torch.long).to(device)
    action_mask = torch.ones(batch_size, action_config.action_horizon, action_config.max_action_dim).to(device)

    action_inputs = BatchFeature(data={
        "state": state,
        "action": action,
        "embodiment_id": embodiment_id,
        "action_mask": action_mask,
    })

    return backbone_inputs, action_inputs


# ──────────────────────────────────────────────────────────────────────
# Backbone forward wrapper
# ──────────────────────────────────────────────────────────────────────

def run_backbone(eagle_model, backbone_inputs):
    """Run the Eagle backbone and return features."""
    keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
    vl_input = {k: backbone_inputs[k] for k in keys_to_use}

    outputs = eagle_model(**vl_input, output_hidden_states=True)
    hidden_states = outputs["hidden_states"][-1]

    image_mask = vl_input["input_ids"] == eagle_model.config.image_token_index
    attention_mask = vl_input["attention_mask"] == 1

    return BatchFeature(data={
        "backbone_features": hidden_states,
        "backbone_attention_mask": attention_mask,
        "image_mask": image_mask,
    })


# ──────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────

def create_fixed_dataset(action_config, num_examples):
    """Create a small fixed dataset of (state, target_action) pairs."""
    torch.manual_seed(123)

    examples = []
    for _ in range(num_examples):
        state = torch.randn(1, 1, action_config.max_state_dim)
        action = torch.randn(1, action_config.action_horizon, action_config.max_action_dim) * 2.0
        embodiment_id = torch.zeros(1, dtype=torch.long)
        action_mask = torch.ones(1, action_config.action_horizon, action_config.max_action_dim)

        examples.append({
            "state": state,
            "action": action,
            "embodiment_id": embodiment_id,
            "action_mask": action_mask,
        })

    return examples


def make_batch(examples, backbone_output, device):
    """Stack dataset examples into a batch and expand backbone features."""
    batch_size = len(examples)
    batch = {
        "state": torch.cat([ex["state"] for ex in examples], dim=0).to(device),
        "action": torch.cat([ex["action"] for ex in examples], dim=0).to(device),
        "embodiment_id": torch.cat([ex["embodiment_id"] for ex in examples], dim=0).to(device),
        "action_mask": torch.cat([ex["action_mask"] for ex in examples], dim=0).to(device),
    }
    action_inputs = BatchFeature(data=batch)

    expanded_backbone = BatchFeature(data={
        "backbone_features": backbone_output["backbone_features"].expand(batch_size, -1, -1).clone(),
        "backbone_attention_mask": backbone_output["backbone_attention_mask"].expand(batch_size, -1).clone(),
        "image_mask": backbone_output["image_mask"].expand(batch_size, -1).clone(),
    })

    return expanded_backbone, action_inputs


@torch.no_grad()
def eval_loss(action_head, backbone_batch, action_batch, num_samples=16):
    """Compute a low-variance eval loss by averaging over multiple timestep samples."""
    action_head.eval()
    total_loss = 0.0
    for i in range(num_samples):
        torch.manual_seed(1000 + i)
        output = action_head(backbone_batch, action_batch)
        total_loss += output["loss"].item()
    action_head.train()
    return total_loss / num_samples


# ──────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────────────────────────────
NUM_DATASET_EXAMPLES = 16
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
EVAL_EVERY = 50
GRAD_ACCUM_STEPS = 4


def main():
    parser = argparse.ArgumentParser(description="Mini GR00T N1.6 training script")
    parser.add_argument("--device", default="neuron", choices=["cpu", "neuron"],
                        help="Device to train on (default: neuron)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of training steps (default: 500)")
    args = parser.parse_args()

    device = args.device
    num_steps = args.steps

    torch.manual_seed(42)

    # ── Build model ──
    eagle_model, action_head, eagle_config, action_config = build_model(device)

    # ── Cache backbone features (frozen) ──
    print("\n" + "=" * 60)
    print("Caching backbone features (backbone is frozen)")
    print("=" * 60)
    backbone_inputs, _ = create_dummy_inputs(eagle_config, action_config, device)
    with torch.no_grad():
        backbone_output = run_backbone(eagle_model, backbone_inputs)

    backbone_output = BatchFeature(data={
        k: v.detach() for k, v in backbone_output.items()
    })
    print(f"    backbone_features: {backbone_output['backbone_features'].shape}")

    del eagle_model
    del backbone_inputs

    # ── Create fixed dataset ──
    print("\n" + "=" * 60)
    print(f"Creating fixed dataset ({NUM_DATASET_EXAMPLES} examples)")
    print("=" * 60)
    dataset = create_fixed_dataset(action_config, NUM_DATASET_EXAMPLES)
    for i, ex in enumerate(dataset):
        print(f"    Example {i}: state={ex['state'].shape}, action={ex['action'].shape}")

    # ── Set up optimizer ──
    action_head.train()
    optimizer = torch.optim.AdamW(
        action_head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    n_trainable = sum(p.numel() for p in action_head.parameters() if p.requires_grad)
    print(f"\n    Trainable parameters: {n_trainable:,}")
    print(f"    Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Grad accum steps: {GRAD_ACCUM_STEPS}")
    print(f"    Steps: {num_steps}")

    # ── Measure initial eval loss ──
    backbone_batch, action_batch = make_batch(dataset, backbone_output, device)
    eval_loss_0 = eval_loss(action_head, backbone_batch, action_batch)

    # ── Training loop ──
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    print(f"{'Step':>6}  {'Train Loss':>12}  {'Eval Loss':>12}  {'Time (ms)':>10}")
    print("-" * 48)
    print(f"{'0':>6}  {'---':>12}  {eval_loss_0:>12.6f}  {'---':>10}")

    train_losses = []
    eval_losses = [eval_loss_0]
    t_total_start = time.time()

    for step in range(1, num_steps + 1):
        t_step = time.time()

        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            backbone_batch, action_batch = make_batch(dataset, backbone_output, device)
            output = action_head(backbone_batch, action_batch)
            loss = output["loss"] / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()

        optimizer.step()

        step_ms = (time.time() - t_step) * 1000
        train_losses.append(accum_loss)

        if step % EVAL_EVERY == 0 or step == num_steps:
            bb_eval, ab_eval = make_batch(dataset, backbone_output, device)
            e_loss = eval_loss(action_head, bb_eval, ab_eval)
            eval_losses.append(e_loss)
            print(f"{step:>6}  {accum_loss:>12.6f}  {e_loss:>12.6f}  {step_ms:>10.1f}")
        else:
            print(f"{step:>6}  {accum_loss:>12.6f}  {'':>12}  {step_ms:>10.1f}")

    t_total = time.time() - t_total_start

    # ── Results ──
    print("\n" + "=" * 60)
    print("Training results")
    print("=" * 60)

    initial_eval = eval_losses[0]
    final_eval = eval_losses[-1]
    reduction_pct = (1 - final_eval / initial_eval) * 100
    decreased = final_eval < initial_eval

    print(f"    Initial eval loss:   {initial_eval:.6f}")
    print(f"    Final eval loss:     {final_eval:.6f}")
    print(f"    Loss reduction:      {reduction_pct:+.1f}%")
    print(f"    Eval loss history:   {['%.4f' % e for e in eval_losses]}")
    print(f"    Train loss (first):  {train_losses[0]:.6f}")
    print(f"    Train loss (last):   {train_losses[-1]:.6f}")
    print(f"    Total time:          {t_total:.2f}s")
    print(f"    Avg step time:       {t_total / num_steps * 1000:.1f}ms")

    if decreased:
        print(f"\n    PASS - Eval loss decreased: "
              f"{initial_eval:.6f} -> {final_eval:.6f} ({reduction_pct:+.1f}%)")
    else:
        print(f"\n    FAIL - Eval loss did not decrease: "
              f"{initial_eval:.6f} -> {final_eval:.6f}")
        sys.exit(1)

    # ── Post-training inference check ──
    print("\n" + "=" * 60)
    print("Post-training inference check")
    print("=" * 60)
    action_head.eval()
    with torch.no_grad():
        test_action_input = BatchFeature(data={
            "state": dataset[0]["state"].to(device),
            "action": dataset[0]["action"].to(device),
            "embodiment_id": dataset[0]["embodiment_id"].to(device),
            "action_mask": dataset[0]["action_mask"].to(device),
        })
        bb_test, _ = make_batch([dataset[0]], backbone_output, device)
        inference_output = action_head.get_action(bb_test, test_action_input)
        action_pred = inference_output["action_pred"]
        target_action = dataset[0]["action"].to(device)

        mse = torch.mean((action_pred - target_action) ** 2).item()
        print(f"    Predicted action shape: {action_pred.shape}")
        print(f"    Target action sample:    {target_action[0, 0, :3].tolist()}")
        print(f"    Predicted action sample: {action_pred[0, 0, :3].tolist()}")
        print(f"    MSE (pred vs target):    {mse:.6f}")

    print("\n" + "=" * 60)
    print(f"SUCCESS - Training loop completed on {device}")
    print("=" * 60)


if __name__ == "__main__":
    main()
