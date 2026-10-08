"""Explicitly restore root policy and nested reference in a PEFT checkpoint.

Transformers 4.57's generic multi-adapter loader skips the root adapter when a
`ref/` subdirectory exists. Both adapters must be restored and checked here.
"""
from pathlib import Path
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file


def verify_adapter(model, directory: Path, adapter_name: str) -> int:
    tensors = load_file(str(directory / "adapter_model.safetensors"))
    checked = 0
    marker = f".{adapter_name}."
    for name, param in model.named_parameters():
        if marker not in name:
            continue
        key = name.replace(marker, ".")
        if key not in tensors:
            raise RuntimeError("checkpoint lacks tensor " + key)
        expected = tensors[key].to(device=param.device, dtype=param.dtype)
        if not torch.equal(param.detach(), expected):
            raise RuntimeError("checkpoint adapter mismatch: " + name)
        checked += 1
    if checked != len(tensors) or not checked:
        raise RuntimeError("incomplete checkpoint tensor verification")
    return checked


def restore_adapters(model, checkpoint: Path, sft_adapter: Path) -> dict:
    policy = load_file(str(checkpoint / "adapter_model.safetensors"))
    reference = load_file(str(checkpoint / "ref/adapter_model.safetensors"))
    set_peft_model_state_dict(model, policy, adapter_name="default")
    set_peft_model_state_dict(model, reference, adapter_name="ref")
    model.set_adapter("default")
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad_(False)
        elif param.requires_grad and ".default." not in name:
            raise RuntimeError("unexpected trainable parameter after resume: " + name)
    policy_count = verify_adapter(model, checkpoint, "default")
    ref_count = verify_adapter(model, checkpoint / "ref", "ref")
    verify_adapter(model, sft_adapter, "ref")
    return {"policy_tensors_equal_to_checkpoint": policy_count,
            "reference_tensors_equal_to_checkpoint_and_original_sft": ref_count,
            "active_adapter": model.active_adapter, "reference_frozen": True}
