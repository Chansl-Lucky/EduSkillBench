#!/usr/bin/env python3
"""Continue the frozen SFT246 LoRA with Skill-free, outcome-based GRPO.

No tests, answer text, rubric or Skill are passed to the rollout policy. TRL
0.29.1 copies the loaded default adapter to a frozen `ref` adapter for SFT KL.
The actual initial tensors and reference tensors are verified before training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from rlvr.sft_grpo_reward import OutcomeJudge, JUDGE, API, append, save, sha

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def load_training(release: Path, test_cases: Path) -> tuple[list[dict], dict]:
    raw = (release / "train.jsonl").read_bytes()
    originals = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    tests = json.loads(test_cases.read_text())
    test_ids = {t["task_id"] for t in tests}
    test_questions = {"".join((t.get("context", "") + t.get("user_prompt", "")).split()) for t in tests}
    rows, seen, source_hashes = [], set(), {}
    for row in originals:
        tid = row["task_id"]
        if tid in seen or tid in test_ids:
            raise ValueError("duplicate/training-test task id overlap: " + tid)
        seen.add(tid)
        messages = row["messages"]
        if row["student_condition"] != "no_skill" or [m["role"] for m in messages] != ["system", "user", "assistant"]:
            raise ValueError("unexpected SFT student prompt: " + tid)
        source = Path(row["source_record"])
        digest = sha(source.read_bytes())
        if digest != row["source_record_sha256"]:
            raise ValueError("frozen source changed: " + tid)
        contract = json.loads(source.read_text())["candidate"]
        if contract["question"] != messages[1]["content"] or contract["answer"] != messages[2]["content"]:
            raise ValueError("frozen source does not match SFT data: " + tid)
        if "".join(contract["question"].split()) in test_questions:
            raise ValueError("exact training-test question overlap: " + tid)
        rubric = contract["rubric"]
        if not rubric or any(not r.get("criterion") or not r.get("verification") for r in rubric):
            raise ValueError("incomplete fixed training rubric: " + tid)
        if len({r["criterion"] for r in rubric}) != len(rubric):
            raise ValueError("duplicate training criterion: " + tid)
        if "<skill>" in json.dumps(messages[:2]).lower():
            raise ValueError("Skill leaked into no-Skill rollout input")
        rows.append({"prompt": messages[:2], "task_id": tid, "skill_id": row["skill_id"],
                     "question": contract["question"], "rubric_json": json.dumps(rubric, ensure_ascii=False),
                     "reference": contract["answer"]})
        source_hashes[str(source)] = digest
    if len(rows) != 246 or len(Counter(r["skill_id"] for r in rows)) != 14:
        raise ValueError("expected the unchanged SFT246 release and all 14 Skills")
    audit = {"train_data_sha256": sha(raw), "train_questions": len(rows),
             "per_skill": dict(Counter(r["skill_id"] for r in rows)), "source_hashes": source_hashes,
             "test_cases_sha256": sha(test_cases.read_bytes()), "test_cases_used_for_training": 0,
             "exact_test_overlap": 0, "semantic_independence_claim": False,
             "rubric_source": "hash-verified frozen SFT source_record; not mutable candidates directory"}
    return rows, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=ROOT / "artifacts/edubench_glm_sft_20261002")
    parser.add_argument("--model", default="${MODEL_ROOT}/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--generation-batch", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--stop-after", type=int, default=0, help="one-step pressure probe, then resume same schedule")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recovery-revision", type=Path, help="audited code-only recovery authorization")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / "train.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows, audit = load_training(args.release, ROOT / ".local-artifacts/handoff_5a9c6c2/data/exports/eduskillbench-305-20261003/cases.json")
    adapter = args.release / "sft/final_adapter"
    if args.generation_batch % args.num_generations or args.generation_batch % args.micro_batch:
        raise ValueError("generation batch must divide into full GRPO groups and micro batches")
    max_steps = math.ceil(len(rows) / (args.generation_batch // args.num_generations))
    config_summary = {"at": now(), "base_model": args.model, "sft_adapter": str(adapter),
        "sft_adapter_sha256": sha((adapter / "adapter_model.safetensors").read_bytes()),
        "condition": "no_skill", "training_target": "final educational artifact; not agent tool routing",
        "judge": JUDGE, "api": os.environ.get("ARK_JUDGE_BASE_URL", API), "reward": "equal fraction of fixed training rubric PASS, valid candidate line ids",
        "reference_policy": "frozen copy of the SFT adapter, not pre-SFT Base", "max_steps": max_steps,
        "unique_prompts": len(rows), "micro_batch": args.micro_batch, "generation_batch": args.generation_batch,
        "num_generations": args.num_generations, "planned_rollouts": len(rows) * args.num_generations,
        "max_completion_length": args.max_completion_length, "learning_rate": args.learning_rate,
        "beta": args.beta, "seed": args.seed, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_code_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in [Path(__file__), Path(__file__).with_name("sft_grpo_reward.py"), Path(__file__).with_name("sft_grpo_checkpoint.py")]}}
    old = out / "manifest.json"
    if old.exists():
        previous = json.loads(old.read_text())
        for key in ("sft_adapter_sha256", "max_steps", "generation_batch", "num_generations", "learning_rate", "beta", "max_completion_length", "seed", "condition", "judge"):
            if previous[key] != config_summary[key]:
                raise ValueError("cannot change frozen run configuration: " + key)
        if previous["source_code_sha256"] != config_summary["source_code_sha256"]:
            if not args.recovery_revision:
                raise ValueError("changed source needs a recorded recovery authorization")
            revision = json.loads(args.recovery_revision.read_text())
            if revision['original_source_sha256'] != previous['source_code_sha256'] or revision['recovered_source_sha256'] != config_summary['source_code_sha256']:
                raise ValueError("recovery source hash authorization mismatch")
            if revision['original_manifest_sha256'] != sha(old.read_bytes()):
                raise ValueError("original run manifest changed")
            save(out / 'active_recovery_revision.json', {**revision, 'authorized_at': now()})
    else:
        save(old, config_summary)
    save(out / "active_runtime.json", {"at": now(), "judge": JUDGE,
         "api": config_summary["api"], "transport_trust_env": False,
         "source_code_sha256": config_summary["source_code_sha256"],
         "gateway_sha256": sha(Path(__file__).with_name("judge.py").read_bytes())})
    save(out / "data_audit.json", audit)
    # Save the frozen inputs, including privileged Judge-only fields, for reproducibility.
    save(out / "training_contracts.json", rows)
    if args.preflight_only:
        print(json.dumps(config_summary, ensure_ascii=False), flush=True)
        return

    import torch
    import transformers
    import trl
    import peft
    from datasets import Dataset
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    key = os.environ.get("ARK_API_KEY")
    if not key:
        raise RuntimeError("ARK_API_KEY is required; do not store it in the run manifest")
    started = time.time()

    def event(kind, **kw):
        item = {"at": now(), "event": kind, **kw}
        append(out / "events.jsonl", item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    event("loading_sft", versions={"torch": torch.__version__, "transformers": transformers.__version__,
                                   "trl": trl.__version__, "peft": peft.__version__})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = [len(tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True)) for r in rows]
    save(out / "prompt_token_audit.json", {"max_prompt_tokens": max(lengths), "mean_prompt_tokens": sum(lengths)/len(lengths),
                                        "truncated_prompts": 0, "rubric_or_answer_in_policy_prompt": False})
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa", device_map={"": 0})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True, adapter_name="default")
    config = GRPOConfig(output_dir=str(out / "checkpoint"), max_steps=max_steps,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=args.generation_batch // args.micro_batch,
        generation_batch_size=args.generation_batch, num_generations=args.num_generations,
        max_completion_length=args.max_completion_length, temperature=0.7, top_p=0.9,
        learning_rate=args.learning_rate, beta=args.beta, lr_scheduler_type="cosine", warmup_steps=5,
        max_grad_norm=0.5, bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, disable_dropout=True,
        use_liger_kernel=True, mask_truncated_completions=True, loss_type="dapo", scale_rewards="group",
        logging_steps=1, logging_first_step=True, save_steps=1 if args.recovery_revision else 10, save_total_limit=3,
        save_only_model=False, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, disable_tqdm=True)

    async def outcome_reward(completions, completion_ids, task_id, question, rubric_json, reference,
                             trainer_state, **unused):
        import asyncio
        gateway = OutcomeJudge(key, out / "judge_cache", concurrency=args.generation_batch)

        async def one(i):
            parts = completions[i]
            text = parts[-1].get("content", "") if isinstance(parts, list) else str(parts)
            # Distinguish actual EOS from exhausting the cap; capped answers are not positive trajectories.
            truncated = bool(completion_ids[i] and completion_ids[i][-1] not in tokenizer.eos_token_id_list) if hasattr(tokenizer, "eos_token_id_list") else bool(completion_ids[i] and completion_ids[i][-1] != tokenizer.eos_token_id)
            record = {"at": now(), "step": trainer_state.global_step, "task_id": task_id[i], "completion": text,
                      "completion_tokens": len(completion_ids[i]), "truncated": truncated}
            if not text.strip() or truncated:
                record.update(reward=0.0, status="truncated" if truncated else "empty")
            else:
                result = await gateway.score(question[i], text, json.loads(rubric_json[i]), reference[i])
                record.update(reward=result["reward"], status="ok", judge_key=result["key"], verdict=result["verdict"])
            append(out / "reward_trace.jsonl", record)
            return record["reward"]
        # Persist all answers + exact sampled token IDs BEFORE invoking any external Judge.
        # A malformed verdict must not destroy an expensive rollout batch.
        save(out / "rollout_batches" / f"step_{trainer_state.global_step}.json", {
            "step": trainer_state.global_step, "at": now(), "task_id": task_id,
            "completions": completions, "completion_ids": completion_ids,
            "question": question, "rubric_json": rubric_json, "reference": reference})
        event("rollouts_ready", step=trainer_state.global_step, count=len(completions))
        try:
            scored = await asyncio.gather(*(one(i) for i in range(len(completions))), return_exceptions=True)
            errors = [e for e in scored if isinstance(e, BaseException)]
            if errors:
                # Let all independent requests finish and retain their caches before surfacing the error.
                raise errors[0]
            rewards = scored
        finally:
            await gateway.close()
        groups = [rewards[i:i+args.num_generations] for i in range(0, len(rewards), args.num_generations)]
        event("reward_batch", step=trainer_state.global_step, rewards=rewards,
              zero_variance_groups=sum(max(g)==min(g) for g in groups), groups=len(groups))
        return rewards

    class Progress(TrainerCallback):
        def on_train_begin(self, args_, state, control, model=None, optimizer=None, lr_scheduler=None, **kw):
            # Trainer reloads save_steps=10 from old TrainerState AFTER computing
            # current args. Apply the authorized safety cadence to the state too.
            if args.recovery_revision:
                state.save_steps = 1
            if resume:
                from rlvr.sft_grpo_checkpoint import verify_adapter
                cp = Path(resume)
                state_on_disk = json.loads((cp/'trainer_state.json').read_text())
                if state.global_step != state_on_disk['global_step']:
                    raise RuntimeError('trainer global_step did not resume')
                verify_adapter(model, cp, 'default')
                verify_adapter(model, adapter, 'ref')
                opt_steps = [int(s['step']) for s in optimizer.state.values() if 'step' in s]
                scheduler_step = lr_scheduler.state_dict().get('last_epoch')
                if not opt_steps or min(opt_steps) != state.global_step or max(opt_steps) != state.global_step:
                    raise RuntimeError('optimizer moment/step state did not resume')
                if scheduler_step != state.global_step:
                    raise RuntimeError('learning-rate schedule did not resume')
                recovery_state = {'at': now(), 'checkpoint': resume, 'step': state.global_step,
                    'policy_matches_checkpoint': True, 'reference_matches_original_sft': True,
                    'optimizer_state_parameters': len(opt_steps), 'optimizer_step': min(opt_steps),
                    'scheduler_last_epoch': scheduler_step, 'rng_state_exists': (cp/'rng_state.pth').exists(),
                    'checkpoint_save_steps': state.save_steps}
                save(out/'resume_audit.json', recovery_state)
                save(out/'status.json', {**recovery_state, 'phase': 'resumed_training', 'max_steps': max_steps})
                event('resume_state_verified', **{k:v for k,v in recovery_state.items() if k!='at'})
        def on_log(self, args_, state, control, logs=None, **kw):
            data = {"at": now(), "phase": "training", "step": state.global_step, "max_steps": max_steps,
                    "wall_seconds_this_process": round(time.time()-started, 2), "metrics": logs,
                    "cuda_peak_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30, 2),
                    "cuda_peak_reserved_gib": round(torch.cuda.max_memory_reserved()/2**30, 2)}
            save(out / "status.json", data); append(out / "train_steps.jsonl", data)
        def on_step_end(self, args_, state, control, **kw):
            if args.recovery_revision:
                control.should_save = True
            if args.stop_after and state.global_step >= args.stop_after:
                control.should_save = True
                control.should_training_stop = True
            return control

    class ResumeSafeGRPOTrainer(GRPOTrainer):
        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            from rlvr.sft_grpo_checkpoint import restore_adapters
            restored = restore_adapters(model or self.model, Path(resume_from_checkpoint), adapter)
            event('checkpoint_policy_and_reference_loaded', checkpoint=str(resume_from_checkpoint), **restored)

    trainer = ResumeSafeGRPOTrainer(model=model, reward_funcs=outcome_reward, args=config,
                          train_dataset=Dataset.from_list(rows), processing_class=tokenizer, callbacks=[Progress()])
    # Prove exact initial policy+reference equality to the original SFT tensors.
    disk = load_file(str(adapter / "adapter_model.safetensors"))
    checked, ref_checked = 0, 0
    for name, param in model.named_parameters():
        if ".default." in name:
            disk_name = name.replace(".default.", ".")
            expected = disk[disk_name].to(device=param.device, dtype=param.dtype)
            if not torch.equal(param.detach(), expected):
                raise RuntimeError("SFT initialization mismatch: " + name)
            ref = model.get_parameter(name.replace(".default.", ".ref."))
            if not torch.equal(param.detach(), ref.detach()) or ref.requires_grad:
                raise RuntimeError("reference is not a frozen SFT copy: " + name)
            checked += 1; ref_checked += 1
        elif param.requires_grad:
            raise RuntimeError("unexpected trainable non-policy parameter: " + name)
    if checked == 0 or checked != len(disk):
        raise RuntimeError("incomplete SFT adapter initialization check")
    del expected, disk
    save(out / "initialization_audit.json", {"policy_tensors_exactly_equal_to_sft": checked,
        "frozen_reference_tensors_exactly_equal_to_sft": ref_checked,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "base_weights_frozen": True, "reference": "SFT", "gpu": torch.cuda.get_device_name(0)})
    event("initialization_verified", tensors=checked, micro_batch=args.micro_batch)
    resume = None
    if args.resume:
        checkpoints = sorted((out / "checkpoint").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            resume = str(checkpoints[-1]); event("resume", checkpoint=resume)
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        # Reusing an already trained adapter means final default weights include SFT+GRPO; no merge needed.
        final = out / ("probe_adapter" if args.stop_after else "final_adapter")
        model.save_pretrained(final, selected_adapters=["default"])
        tokenizer.save_pretrained(final)
        metrics = {**result.metrics, "global_step": trainer.state.global_step,
            "elapsed_wall_seconds": time.time()-started, "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
            "adapter_sha256": sha((final / "adapter_model.safetensors").read_bytes()), "adapter": str(final)}
        save(out / ("probe_metrics.json" if args.stop_after else "train_metrics.json"), metrics)
        phase = "pressure_probe_complete" if args.stop_after else "finished"
        save(out / "status.json", {"at": now(), "phase": phase, **metrics})
        event(phase, **metrics)
        if not args.stop_after:
            save(out / "FINISHED.json", {"at": now(), "phase": "trained_not_yet_test_evaluated", **metrics})
    except Exception as exc:
        failed_status = {"at": now(), "phase": "failed", "step": trainer.state.global_step,
            "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        save(out / "status.json", failed_status)
        append(out / 'failure_history.jsonl', failed_status)
        raise


if __name__ == "__main__":
    main()
