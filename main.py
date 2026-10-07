import os
import torch

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
)
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Value
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from collections import Counter
import numpy as np
import math
import glob


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


def prepare_dataset_for_generation(dataset, max_prompt_chars=1500, test_size=0.2):
    # Filter very long prompts
    dataset = dataset.filter(lambda x: len(x["prompt"]) < max_prompt_chars)
    print(f"After filtering long prompts: {len(dataset):,} rows")

    # Keep only needed columns
    dataset = dataset.remove_columns(["for_devs", "contributor"])

    # Split
    split = dataset.train_test_split(test_size=test_size, seed=42)
    print(f"Train: {len(split['train']):,}  |  Val: {len(split['test']):,}")
    return split["train"], split["test"]


def tokenize_for_generation(dataset, tokenizer, act_col, prompt_col, max_length=512):
    def _build_prompt(act):
        return f"act: {act}\nprompt:"

    def _build_target(prompt):
        return f" {prompt}{tokenizer.eos_token}"

    def _tokenize(batch):
        input_texts = [_build_prompt(a) for a in batch[act_col]]
        target_texts = [_build_target(p) for p in batch[prompt_col]]
        full_texts = [i + t for i, t in zip(input_texts, target_texts)]

        # Tokenize full text (with truncation)
        enc = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Tokenize just the input part to find its length
        enc_input = tokenizer(
            input_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )

        # Build labels: mask input tokens, keep target tokens
        labels = []
        for ids, input_ids in zip(enc["input_ids"], enc_input["input_ids"]):
            n_input = min(len(input_ids), len(ids))
            lab = [-100] * n_input + ids[n_input:]
            labels.append(lab)

        enc["labels"] = labels
        return enc

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing for generation",
    )


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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        warmup_steps=10,
        lr_scheduler_type="linear",
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


def generate_prompt(peft_model, tokenizer, act, max_new_tokens=200):
    peft_model.eval()
    prompt = f"act: {act}\nprompt:"
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

    with torch.no_grad():
        out = peft_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Cut off the input part
    generated = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_trained_model(base_model, peft_path):
    peft_model = PeftModel.from_pretrained(base_model, peft_path)
    peft_model.eval()
    return peft_model


# ============================================================
# 4. evaluation
# ============================================================
def compute_perplexity(trainer):
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"])
    print(f"Validation Perplexity: {ppl:.2f}")
    return ppl


def compare_with_baseline(base_model, peft_model, tokenizer, test_acts):
    print("\n" + "=" * 70)
    print("BASELINE vs PROMPT-TUNED")
    print("=" * 70)
    for act in test_acts:
        print(f"\n--- ACT: {act} ---")
        print(
            "[BASE]     ",
            generate_prompt(base_model, tokenizer, act, max_new_tokens=100)[:200],
        )
        print(
            "[TUNED]    ",
            generate_prompt(peft_model, tokenizer, act, max_new_tokens=100)[:200],
        )


if __name__ == "__main__":
    model_name = "bigscience/bloomz-1b1"
    NUM_VIRTUAL_TOKEN = 100
    NUM_EPOCHS = 8
    OUTPUT_DIR_PROMPT = "./data/prompt_data"
    FINAL_SAVE_PATH = f"{OUTPUT_DIR_PROMPT}/final_peft"

    # -------- If final model exists, skip everything --------
    if os.path.exists(FINAL_SAVE_PATH):
        print(f"Final model already exists at {FINAL_SAVE_PATH}. Loading it...")
        base_model, base_tokenizer = load_base_model_and_tokenizer(model_name)
        peft_model = load_trained_model(base_model, FINAL_SAVE_PATH)

        for act in ["Ethereum Developer", "Travel Guide"]:
            print(f"\n--- {act} ---")
            print(generate_prompt(peft_model, base_tokenizer, act))
        exit()

    # -------- Load base --------
    base_model, base_tokenizer = load_base_model_and_tokenizer(model_name)

    # -------- Dataset --------
    dataset = load_dataset("fka/prompts.chat")
    # dataset_first_look(dataset)

    train_ds, val_ds = prepare_dataset_for_generation(
        dataset["train"], max_prompt_chars=1500, test_size=0.1
    )
    tokenized_train = tokenize_for_generation(train_ds, base_tokenizer, "act", "prompt", 512)
    tokenized_val = tokenize_for_generation(val_ds, base_tokenizer, "act", "prompt", 512)

    # -------- PEFT --------
    peft_config = create_peft_config(model_name, NUM_VIRTUAL_TOKEN)
    peft_model = build_peft_model(base_model, peft_config)
    peft_model.enable_input_require_grads()

    # -------- Trainer --------
    training_args = create_training_arguments(
        OUTPUT_DIR_PROMPT, NUM_EPOCHS, learning_rate=3e-4
    )
    trainer = create_trainer(
        peft_model, base_tokenizer, training_args, tokenized_train, tokenized_val
    )

    # -------- Resume if a checkpoint exists --------
    checkpoints = sorted(glob.glob(f"{OUTPUT_DIR_PROMPT}/checkpoint-*"))
    if checkpoints:
        print(f"Resuming from {checkpoints[-1]}")
        trainer.train(resume_from_checkpoint=checkpoints[-1])
    else:
        print("Starting fresh training.")
        trainer.train()

    # -------- Save final --------
    peft_model.save_pretrained(FINAL_SAVE_PATH)
    base_tokenizer.save_pretrained(FINAL_SAVE_PATH)
    print(f"Saved to {FINAL_SAVE_PATH}")

    # -------- Evaluate --------
    compute_perplexity(trainer)

    compare_with_baseline(
        base_model,
        peft_model,
        base_tokenizer,
        ["Ethereum Developer", "Linux Terminal", "Travel Guide"],
    )

    for act in ["Ethereum Developer", "Linux Terminal", "Travel Guide"]:
        print(f"\n--- {act} ---")
        print(generate_prompt(peft_model, base_tokenizer, act))
