import os
import torch
import re
import json

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Value
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from collections import Counter
import numpy as np
import math
import glob
from functools import partial
from save_peft_model import list_checkpoints, pick_best_checkpoint, save_peft_from_checkpoint
import gc


# ============================================================
# 1. base model
# ============================================================
def load_base_model_and_tokenizer(model_name):
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return original_model, tokenizer


def get_outputs(model, tokenizer, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
    )
    return outputs


def zero_shot(model, tokenizer, prompt):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = get_outputs(model, tokenizer, tokenized_prompt)
    output_ids = output[0][tokenized_prompt["input_ids"].shape[1] :]
    model_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    result = f"""
    prompt: {prompt}
    model answer: {model_output}"""

    return result


# ============================================================
# 2. dataset
# ============================================================
def loading_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


def dataset_first_look(dataset, sample_size=2, text_columns=None):
    # ---------- Normalize input ----------
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    if not isinstance(dataset, DatasetDict):
        raise TypeError("Input must be a Dataset or DatasetDict.")

    print("\n" + "=" * 70)
    print("DATASET FIRST LOOK")
    print("=" * 70)

    # ---------- 1) Splits ----------
    print("\n[1] SPLITS")
    print(f"    Number of splits: {len(dataset)}")
    for split_name, split_data in dataset.items():
        print(f"    - {split_name}: {len(split_data):,} rows")

    total_rows = sum(len(d) for d in dataset.values())
    print(f"    -> Total rows: {total_rows:,}")

    # Use the first split for structural analysis
    first_split_name = list(dataset.keys())[0]
    first_split = dataset[first_split_name]

    # ---------- 2) Column names ----------
    print(f"\n[2] COLUMNS (based on split: {first_split_name})")
    print(f"    {first_split.column_names}")

    # ---------- 3) Column types ----------
    print(f"\n[3] COLUMN TYPES")
    features = first_split.features
    for col_name, feat in features.items():
        print(f"    - {col_name}: {feat}")

    # ---------- 4) Unique values per column ----------
    print(f"\n[4] UNIQUE VALUES PER COLUMN")
    for col_name in first_split.column_names:
        try:
            values = first_split[col_name]
            unique_vals = set(values)
            n_unique = len(unique_vals)

            if n_unique <= 20:
                counter = Counter(values)
                print(f"    - {col_name}: {n_unique} unique values")
                for val, count in counter.most_common():
                    pct = 100 * count / len(values)
                    print(f"        * {val!r}: {count:,} ({pct:.1f}%)")
            else:
                print(f"    - {col_name}: {n_unique:,} unique values")
        except Exception as e:
            print(f"    - {col_name}: could not be computed ({e})")

    # ---------- 5) Null / missing values ----------
    print(f"\n[5] NULL / MISSING VALUES")
    for col_name in first_split.column_names:
        try:
            values = first_split[col_name]
            n_null = sum(1 for v in values if v is None or v == "")
            pct = 100 * n_null / len(values) if len(values) else 0
            status = "OK " if n_null == 0 else "WARN"
            print(f"    [{status}] {col_name}: {n_null:,} missing ({pct:.2f}%)")
        except Exception as e:
            print(f"    [ERR] {col_name}: could not be checked ({e})")

    # ---------- 6) General info ----------
    print(f"\n[6] GENERAL INFO")
    print(f"    Total splits: {len(dataset)}")
    print(f"    Total rows: {total_rows:,}")
    print(f"    Total columns: {len(first_split.column_names)}")
    print(f"    Columns: {first_split.column_names}")

    # ---------- 7) Dataset size estimate ----------
    print(f"\n[7] SIZE ESTIMATE")
    try:
        for split_name, split_data in dataset.items():
            size_bytes = split_data.size_in_bytes
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024
            if size_gb >= 1:
                print(f"    - {split_name}: {size_gb:.2f} GB")
            else:
                print(f"    - {split_name}: {size_mb:.2f} MB")
    except Exception as e:
        print(f"    Could not be computed: {e}")

    # ---------- 8) Text length distribution ----------
    print(f"\n[8] TEXT LENGTH DISTRIBUTION")

    if text_columns is None:
        text_columns = []
        for col_name, feat in features.items():
            if isinstance(feat, Value) and feat.dtype == "string":
                text_columns.append(col_name)

    if not text_columns:
        print("    No text column detected.")
    else:
        for col_name in text_columns:
            try:
                values = first_split[col_name]
                lengths = np.array([len(str(v)) for v in values])
                print(f"    - {col_name}:")
                print(f"        * min:    {lengths.min()}")
                print(f"        * max:    {lengths.max()}")
                print(f"        * mean:   {lengths.mean():.1f}")
                print(f"        * median: {np.median(lengths):.1f}")
                print(f"        * std:    {lengths.std():.1f}")
            except Exception as e:
                print(f"    - {col_name}: {e}")

    # ---------- 9) Class balance for label-like columns ----------
    print(f"\n[9] CLASS BALANCE (low-cardinality columns)")
    for col_name in first_split.column_names:
        try:
            values = first_split[col_name]
            unique_vals = set(values)
            if 1 < len(unique_vals) <= 20:
                counter = Counter(values)
                counts = list(counter.values())
                imbalance = max(counts) / min(counts) if min(counts) > 0 else float("inf")
                print(f"    - {col_name}: imbalance ratio = {imbalance:.2f}x")
                if imbalance > 5:
                    print(f"        [WARN] Dataset is heavily imbalanced.")
                elif imbalance > 2:
                    print(f"        [NOTE] Dataset is slightly imbalanced.")
                else:
                    print(f"        [OK]   Dataset is roughly balanced.")
        except Exception:
            pass

    # ---------- 10) Samples ----------
    print(f"\n[10] SAMPLES")
    for split_name, split_data in dataset.items():
        n_samples = min(sample_size, len(split_data))
        print(f"\n    --- {split_name} ---")
        for i in range(n_samples):
            sample = split_data[i]
            print(f"    Sample {i + 1}:")
            for k, v in sample.items():
                v_str = str(v)
                if len(v_str) > 200:
                    v_str = v_str[:200] + "..."
                print(f"        {k}: {v_str!r}")
            print()

    print("=" * 70)
    print("Analysis finished.")
    print("=" * 70)


def preprocess_function(examples, tokenizer, max_length=512):
    full_texts = [
        tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=False)
        for msg_list in examples["messages"]
    ]

    prompt_texts = [
        tokenizer.apply_chat_template(
            msg_list[:-1], tokenize=False, add_generation_prompt=True
        )
        for msg_list in examples["messages"]
    ]

    tokenized = tokenizer(full_texts, truncation=True, max_length=max_length, padding=False)
    prompt_tokenized = tokenizer(
        prompt_texts, truncation=True, max_length=max_length, padding=False
    )

    labels = []
    for ids, prompt_ids in zip(tokenized["input_ids"], prompt_tokenized["input_ids"]):
        n_prompt = min(len(prompt_ids), len(ids))
        label = [-100] * n_prompt + ids[n_prompt:]
        labels.append(label)

    tokenized["labels"] = labels
    return tokenized


# ============================================================
# 3. prompt tuning
# ============================================================
def create_peft_config(model_name, virtual_token):
    config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=virtual_token,
        tokenizer_name_or_path=model_name,
    )

    return config


def build_peft_model(model, peft_config):
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def create_training_arguments(path, epochs, learning_rate=1e-3):
    training_args = TrainingArguments(
        output_dir=path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
    )
    return training_args


def create_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    return trainer


def generate_prompt(peft_model, tokenizer, user_request, max_new_tokens=600):
    peft_model.eval()

    messages = [
        {"role": "system", "content": "You are a system prompt generator."},
        {"role": "user", "content": user_request},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

    with torch.no_grad():
        out = peft_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.85,
            repetition_penalty=1.4,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_trained_model(base_model, peft_path):
    peft_model = PeftModel.from_pretrained(base_model, peft_path)
    peft_model.eval()
    return peft_model


def ask_resume_or_finalize(output_dir, checkpoints):
    """
    Ask the user what to do when checkpoints exist.
    Returns one of: 'resume', 'finalize', 'quit'
    """
    print("\n" + "=" * 60)
    print("CHECKPOINTS FOUND")
    print("=" * 60)
    for step, path in sorted(
        [(int(re.search(r"checkpoint-(\d+)", c).group(1)), c) for c in checkpoints]
    ):
        print(f"  step {step:>6}  →  {path}")

    print("\nWhat do you want to do?")
    print("  [c] Continue training from the last checkpoint")
    print("  [f] Finalize: use the last checkpoint as the final PEFT model")
    print("  [q] Quit")

    while True:
        choice = input("\nYour choice [c/f/q]: ").strip().lower()
        if choice in ("c", "continue"):
            return "resume"
        elif choice in ("f", "finalize"):
            return "finalize"
        elif choice in ("q", "quit"):
            return "quit"
        else:
            print("Invalid choice. Please enter c, f, or q.")


# ============================================================
# 4. evaluation
# ============================================================
def compute_perplexity(trainer):
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"])
    print(f"Validation Perplexity: {ppl:.2f}")
    return ppl


# def compare_with_baseline(base_model, peft_model, tokenizer, test_acts):
#     print("\n" + "=" * 70)
#     print("BASELINE vs PROMPT-TUNED")
#     print("=" * 70)
#     for act in test_acts:
#         print(f"\n--- ACT: {act} ---")
#         print(
#             "[BASE]     ",
#             generate_prompt(base_model, tokenizer, act, max_new_tokens=100)[:200],
#         )
#         print(
#             "[TUNED]    ",
#             generate_prompt(peft_model, tokenizer, act, max_new_tokens=100)[:200],
#         )


# ============================================================
# 5. fix bugs
# ============================================================
def prepare_resume_checkpoint(trainer):
    """
    Find the latest checkpoint and recreate a missing trainer_state.json.
    Existing trainer_state.json files are never overwritten.
    """
    output_dir = trainer.args.output_dir
    checkpoint = get_last_checkpoint(output_dir)

    if checkpoint is None:
        print("No checkpoint found. Starting fresh.")
        return None

    state_path = os.path.join(checkpoint, "trainer_state.json")

    print(f"Found checkpoint: {checkpoint}")
    print(f"Configured save_steps: {trainer.args.save_steps}")
    print(f"Configured eval_steps: {trainer.args.eval_steps}")

    if os.path.isfile(state_path):
        # Validate that the existing file is valid JSON.
        with open(state_path, "r", encoding="utf-8") as f:
            json.load(f)

        print("Existing trainer_state.json found.")
        return checkpoint

    match = re.search(r"checkpoint-(\d+)$", checkpoint)
    if match is None:
        raise RuntimeError(f"Cannot determine checkpoint step: {checkpoint}")

    step = int(match.group(1))

    # Check that the checkpoint contains model weights.
    has_weights = any(
        os.path.isfile(os.path.join(checkpoint, filename))
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    )

    if not has_weights:
        raise RuntimeError(
            f"No recognized model weight file found in {checkpoint}. "
            "Do not attempt to resume until the checkpoint is inspected."
        )

    # Recreate a minimal training state.
    state = TrainerState(
        global_step=step,
        logging_steps=trainer.args.logging_steps,
        eval_steps=trainer.args.eval_steps,
        save_steps=trainer.args.save_steps,
        num_train_epochs=int(trainer.args.num_train_epochs),
        train_batch_size=trainer.args.train_batch_size,
    )

    control = TrainerControl()

    if hasattr(control, "state"):
        control_state = control.state()
    else:
        from dataclasses import asdict, is_dataclass

        if not is_dataclass(control):
            raise RuntimeError(
                "Cannot determine the expected TrainerControl state "
                "format for this Transformers version."
            )

        control_state = asdict(control)

    callbacks = state.stateful_callbacks

    if isinstance(callbacks, dict):
        callbacks["TrainerControl"] = control_state
    elif isinstance(callbacks, list):
        callbacks.append(
            {
                "args": ["TrainerControl"],
                "attributes": control_state,
            }
        )
    else:
        raise RuntimeError(
            "Unsupported stateful_callbacks format. "
            "Inspect the installed Transformers version before repairing."
        )

    state.save_to_json(state_path)

    print(f"Recreated missing state: {state_path}")
    print(
        "Warning: previous log history and best-model metadata "
        "cannot be recovered from the checkpoint number alone."
    )

    return checkpoint


def ensure_chat_template(tokenizer):
    """Set a chat template if the tokenizer doesn't have one."""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )
        print("Chat template set manually.")
    return tokenizer


if __name__ == "__main__":
    # ==================== CONFIG ====================
    model_name = "bigscience/bloomz-1b1"
    NUM_VIRTUAL_TOKEN = 50
    NUM_EPOCHS = 5
    OUTPUT_DIR_PROMPT = "./data/prompt_data"
    FINAL_SAVE_PATH = f"{OUTPUT_DIR_PROMPT}/final_peft"

    TEST_REQUESTS = [
        "I need system instructions to define a friendly assistant personality: "
        "an authority in Programmer, applications. The instructions must be brief",
        "Create a system prompt for a reliable agent that will help users with "
        "Writer problems. The prompt should be concise.",
    ]

    # ==================== STATE ====================
    peft_model = None
    base_tokenizer = None
    trainer = None
    skip_training = False
    already_saved = False

    # ==================== STEP 0: Load existing final model ====================
    if os.path.exists(FINAL_SAVE_PATH):
        print(f"\n{'=' * 60}")
        print(f"Final model already exists at {FINAL_SAVE_PATH}")
        print(f"Loading for inference only. Delete the folder to retrain.")
        print(f"{'=' * 60}\n")

        base_model, base_tokenizer = load_base_model_and_tokenizer(model_name)
        base_tokenizer = ensure_chat_template(base_tokenizer)
        peft_model = load_trained_model(base_model, FINAL_SAVE_PATH)
        skip_training = True

    # ==================== STEPS 1-6: Training pipeline ====================
    if not skip_training:
        # -------- STEP 1: Load base model --------
        base_model, base_tokenizer = load_base_model_and_tokenizer(model_name)
        base_tokenizer = ensure_chat_template(base_tokenizer)

        # -------- STEP 2: Load dataset --------
        dataset = load_dataset("kth8/system_prompts_Jobs-20000x")
        dataset = dataset.remove_columns(["uuid", "job", "length", "tokens"])

        split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_ds = split_dataset["train"]
        val_ds = split_dataset["test"]
        print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

        # -------- STEP 3: Tokenize --------
        tokenize_fn = partial(preprocess_function, tokenizer=base_tokenizer, max_length=256)
        tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["messages"])
        tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=["messages"])

        # -------- STEP 4: Build PEFT model --------
        peft_config = create_peft_config(model_name, NUM_VIRTUAL_TOKEN)
        peft_model = build_peft_model(base_model, peft_config)
        peft_model.enable_input_require_grads()

        # -------- STEP 5: Build Trainer --------
        training_args = create_training_arguments(OUTPUT_DIR_PROMPT, NUM_EPOCHS)
        trainer = create_trainer(
            peft_model, base_tokenizer, training_args, tokenized_train, tokenized_val
        )

        # -------- STEP 6: Resume / Finalize / Fresh --------
        checkpoints = glob.glob(f"{OUTPUT_DIR_PROMPT}/checkpoint-*")
        checkpoints = sorted(
            checkpoints, key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
        )

        if checkpoints:
            mode = ask_resume_or_finalize(OUTPUT_DIR_PROMPT, checkpoints)

            if mode == "quit":
                print("Exiting without any action.")
                exit()

            elif mode == "finalize":
                last_ckpt = checkpoints[-1]
                print(f"\nUsing checkpoint as final model: {last_ckpt}")

                # Free GPU memory
                del peft_model, base_model, trainer
                gc.collect()
                torch.cuda.empty_cache()

                # Load fresh base + attach checkpoint
                print("Loading fresh base model for finalization...")
                fresh_base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype=torch.float16,
                )
                peft_model, base_tokenizer = save_peft_from_checkpoint(
                    checkpoint_path=last_ckpt,
                    save_path=FINAL_SAVE_PATH,
                    base_model=fresh_base_model,
                    tokenizer=base_tokenizer,
                )
                print(f"✅ Saved to {FINAL_SAVE_PATH}")
                trainer = None
                already_saved = True

            elif mode == "resume":
                print(f"\nResuming from {checkpoints[-1]}")
                trainer.train(resume_from_checkpoint=checkpoints[-1])

        else:
            print("No checkpoints found. Starting fresh training.")
            trainer.train()

    # ==================== STEP 7: Save final model ====================
    if not skip_training and not already_saved:
        peft_model.save_pretrained(FINAL_SAVE_PATH)
        base_tokenizer.save_pretrained(FINAL_SAVE_PATH)
        print(f"Saved to {FINAL_SAVE_PATH}")

    # ==================== STEP 8: Evaluate ====================
    print(f"\n{'=' * 60}")
    print("STEP 8: EVALUATION")
    print(f"{'=' * 60}")

    if trainer is not None:
        compute_perplexity(trainer)
    else:
        print("Skipping perplexity (no trainer available).")

    for req in TEST_REQUESTS:
        print(f"\n{'-' * 50}")
        print(f"USER REQUEST: {req}")
        print(f"{'-' * 50}")
        print("MODEL OUTPUT:")
        print(generate_prompt(peft_model, base_tokenizer, req))
