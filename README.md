# bloomz-system-prompt-generator

<br>

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-Prompt%20Tuning-orange.svg)
![Trainable](https://img.shields.io/badge/Trainable%20Params-0.0072%25-blueviolet.svg)
![Task](https://img.shields.io/badge/Task-System%20Prompt%20Generation-success.svg)
![Perplexity](https://img.shields.io/badge/Perplexity-9.42-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

# 🧠 System Prompt Generation with Prompt Tuning

A parameter-efficient fine-tuning pipeline that adapts **bloomz-1b1** into a **System Prompt generator** — a model that reads a short user request describing a desired persona and produces a structured, role-based System Prompt suitable for an LLM agent.

The project uses **Prompt Tuning** (PEFT), training only **50 soft prompt tokens** (76,800 parameters, **0.0072%** of the model) on a single **8 GB consumer GPU** (RTX 4060 Laptop). The full workflow covers: chat-formatted data loading, chat-template setup, tokenization with prompt-token masking, Prompt Tuning fine-tuning with checkpoint resume, and generation-time evaluation with custom sampling parameters.

**Result:** Validation perplexity drops from **~200 (base model)** to **9.42** after tuning. The tuned model produces structured System Prompts with `Core Responsibilities` sections and role-specific content, while the base model produces ungrounded or irrelevant output.

---

## 🎯 What This Project Does

1. Loads a chat-formatted instruction dataset from Hugging Face.
2. Splits data into **90% train / 10% validation** with a fixed seed.
3. Applies the tokenizer's chat template — or installs one manually if missing.
4. **Masks the prompt tokens** in the labels so the model learns to produce *only* the assistant response.
5. Wraps the base model with **Prompt Tuning** — 50 trainable virtual tokens, everything else frozen.
6. Fine-tunes with checkpoint resume support and an interactive finalize/resume prompt.
7. Evaluates using **validation perplexity** and side-by-side qualitative generation.

---

## 📂 Project Structure

    .
    ├── data/
    │   └── prompt_data/
    │       ├── checkpoint-5000/
    │       ├── checkpoint-6000/
    │       ├── checkpoint-7000/          # PEFT adapter + optimizer + scheduler
    │       └── final_peft/               # Final adapter + tokenizer (ready to load)
    │
    ├── main.py                           # Full pipeline
    ├── save_peft_model.py                # Helper: save best/last checkpoint as PEFT
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## 📊 Dataset

**Source:** [`kth8/system_prompts_Jobs-20000x`](https://huggingface.co/datasets/kth8/system_prompts_Jobs-20000x)

A chat-formatted dataset of user requests paired with generated System Prompts. Each row is a `messages` list in OpenAI-style format:

    [
        {"role": "system",    "content": "You are a system prompt generator."},
        {"role": "user",      "content": "<user's request for a persona>"},
        {"role": "assistant", "content": "<generated System Prompt>"}
    ]

**Metadata columns** (`uuid`, `job`, `length`, `tokens`) are dropped before training.

### Splitting

    train_test_split(test_size=0.1, seed=42)

| Split      | Size    | Percentage |
| ---------- | ------- | ---------- |
| Train      | 18,843  | 90%        |
| Validation |  2,094  | 10%        |

---

## 🧩 Chat Template & Label Masking

`bloomz-1b1` ships **without a chat template**. The pipeline installs a minimal one at runtime:

    {% for message in messages %}
    {{ message['role'] }}: {{ message['content'] }}
    {% endfor %}
    {% if add_generation_prompt %}assistant:{% endif %}

This renders each conversation as:

    system: You are a system prompt generator.
    user: I need a system prompt for a chef who teaches kids to cook
    assistant: <target>

### Why Prompt Masking Matters

Without masking, the model wastes capacity learning to *reproduce* the prompt and user turn. All prompt tokens are replaced with `-100` in the labels tensor, so the loss is computed **only on the assistant's response**:

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

- `-100` is PyTorch's **"ignore in loss"** sentinel.
- This is the single most important step for making a decoder-only LLM behave like an instruction-following assistant.

### Tokenization Settings

| Parameter     | Value | Reason                                        |
| ------------- | ----- | --------------------------------------------- |
| `max_length`  | 256   | Median conversation length fits comfortably   |
| `truncation`  | True  | Safety for long samples                       |
| `padding`     | False | Handled by `DataCollatorForSeq2Seq` later     |

---

## 🧠 Model & Prompt Tuning Configuration

**Base model:** `bigscience/bloomz-1b1` — decoder-only transformer, ~1.07 B parameters, multilingual (46 languages), instruction-tuned.

### Prompt Tuning (PEFT)

    PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=50,
        tokenizer_name_or_path=model_name,
    )

A set of **50 learnable "virtual tokens"** is prepended to the input embeddings. These tokens have no vocabulary entry — they are free parameters optimized during training. Everything else in the model stays frozen.

| Parameter            | Value        | Note                                    |
| -------------------- | ------------ | --------------------------------------- |
| `num_virtual_tokens` | 50           | Prepended soft prompt                   |
| `prompt_tuning_init` | `RANDOM`     | Other options: `TEXT` (init from words) |
| `task_type`          | `CAUSAL_LM`  | Decoder-only LM                         |

**Trainable parameters:** **76,800** (**0.0072%** of the full model). This is roughly **1/130,000** of the base model — a vanishingly small footprint, which is the whole point of Prompt Tuning.

### Why So Few Parameters Still Work

Prompt Tuning is scale-dependent. It works best when the base model already has strong instruction-following ability. `bloomz` is instruction-tuned on `xP3`, so it responds to soft prompts well even at 1B scale.

---

## ⚙️ Training Configuration

| Parameter                       | Value    |
| ------------------------------- | -------- |
| `per_device_train_batch_size`   | 2        |
| `gradient_accumulation_steps`   | 4        |
| **Effective batch size**        | **8**    |
| `learning_rate`                 | 1e-3     |
| `lr_scheduler_type`             | cosine   |
| `warmup_steps`                  | 10       |
| `num_train_epochs`              | 5        |
| `fp16`                          | True     |
| `gradient_checkpointing`        | True     |
| `save_strategy` / `save_steps`  | steps / 1000 |
| `eval_strategy` / `eval_steps`  | steps / 1000 |
| `save_total_limit`              | 3        |
| `load_best_model_at_end`        | True     |

**Hardware:** RTX 4060 Laptop, 8 GB VRAM. A full run takes **~3 hours**.

### Checkpoint Resume & Interactive Finalize

When checkpoints already exist in `OUTPUT_DIR_PROMPT`, the script prompts:

    What do you want to do?
      [c] Continue training from the last checkpoint
      [f] Finalize: use the last checkpoint as the final PEFT model
      [q] Quit

- `c` resumes from the latest checkpoint.
- `f` loads the latest checkpoint, saves it as `final_peft/`, runs the test suite, and exits.
- `q` exits without any action.

The script also detects `final_peft/` at startup — if it exists, training is skipped entirely and only inference runs.

> **Note:** `save_steps` must be a round multiple of `eval_steps` when `load_best_model_at_end=True`. If you edit either after resuming, `trainer_state.json` inside the checkpoint must match — otherwise the Trainer uses stale values.

---

## 💾 Saving & Loading

Only the **adapter weights and tokenizer** are saved (~20 MB total, not the full 1 GB base model):

    data/prompt_data/final_peft/
    ├── adapter_config.json
    ├── adapter_model.safetensors      # ~300 KB — the actual soft prompt
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── chat_template.jinja

### Loading for Inference

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloomz-1b1",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("data/prompt_data/final_peft")
    model = PeftModel.from_pretrained(base_model, "data/prompt_data/final_peft")
    model.eval()

### Standalone Helper Module

`save_peft_model.py` can be imported or used standalone:

    from save_peft_model import save_peft_from_checkpoint

    model, tokenizer = save_peft_from_checkpoint(
        checkpoint_path="data/prompt_data/checkpoint-7000",
        save_path="data/prompt_data/final_peft",
        base_model=base_model,
        tokenizer=tokenizer,
    )

---

## 📊 Evaluation Results

### Quantitative

| Model                            | Val Loss | Perplexity |
| -------------------------------- | -------: | ---------: |
| Base bloomz-1b1 (no tuning)      |    ~5.3  |   **~200** |
| **Prompt-Tuned (checkpoint-7000)** | **2.24** |   **9.42** |
| **Improvement**                  |  **~2.4×** | **~21×** |

Perplexity of **9.42** means the model is, on average, choosing between roughly 9 plausible next tokens at each step — a strong result for a 1B model on a generative task.

### Qualitative — Base vs Tuned

**User request:**
> *"I need system instructions to define a friendly assistant personality: an authority in Programmer, applications. The instructions must be brief."*

**Base model output:**
> *"What is the name of your project? Ethereum"* — irrelevant

**Prompt-tuned output:**
> *"You are a highly skilled Application Developer with deep knowledge of C# and .NET core technologies. Your role is to provide concise, accurate guidance across all phases of project lifecycle..."* — on-task

The gap between base and tuned confirms the soft prompt induced a **genuine behavioral change**, not a marginal gain.

### Generation Parameters

At inference time, the pipeline uses:

    do_sample=True,
    temperature=0.4,
    top_p=0.85,
    repetition_penalty=1.4,
    no_repeat_ngram_size=4,
    max_new_tokens=600,

These settings were chosen empirically to reduce repetition and keep the output focused.

---

## ⚠️ Known Limitations

Prompt Tuning on a 1B model has real ceilings. Being honest about them:

- **Hallucination.** The model invents brand names, "20 years of experience", and unverifiable facts. This is a **base-model capacity** issue — Prompt Tuning cannot inject factual knowledge.
- **Incomplete lists.** Enumerations like `1. ... 2. ... 3. ...` sometimes leave items empty. The model learned the *shape* of a System Prompt but not always the content.
- **Markdown inconsistency.** Bold markers (`**`) are sometimes unbalanced. The training data has varied formatting.
- **No generalization to out-of-distribution personas.** The model performs best on persona categories seen in training.

For production use, a larger base model (3B+) or **LoRA** (which trains ~30M parameters vs. 77K here) would substantially improve output quality.

---

## 🛠️ Main Functions (in `main.py`)

### Data Pipeline

    load_base_model_and_tokenizer()   # base model + tokenizer with pad_token fix
    ensure_chat_template()            # install minimal chat template if missing
    preprocess_function()             # chat-template + prompt-token masking

### Model & Training

    create_peft_config()              # PromptTuningConfig
    build_peft_model()                # wrap base with PEFT
    create_training_arguments()       # TrainingArguments
    create_trainer()                  # Trainer with custom collator
    ask_resume_or_finalize()          # interactive checkpoint decision

### Inference & Evaluation

    generate_prompt()                 # generation with sampling params
    load_trained_model()              # load PEFT adapter for inference
    compute_perplexity()              # validation perplexity

### Helpers (in `save_peft_model.py`)

    list_checkpoints()                # enumerate checkpoint-* dirs
    pick_best_checkpoint()            # read trainer_state.json for best
    save_peft_from_checkpoint()       # export adapter + tokenizer

---

## 🚀 Usage

### 1. Install Dependencies

    pip install -r requirements.txt

For CUDA 12.1 (RTX 4060):

    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

### 2. Train

    python main.py

The pipeline runs end-to-end:

    Load dataset → Split 90/10 → Apply chat template
        → Tokenize with label masking
        → Build PromptTuningConfig (50 virtual tokens)
        → Fine-tune with Trainer (resume-aware)
        → Save adapter to final_peft/
        → Evaluate perplexity + run test requests

### 3. Inference Only

If `final_peft/` exists, `main.py` automatically skips training and runs inference. To re-train, delete the folder:

    rm -rf data/prompt_data/final_peft

### 4. Quick Test

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloomz-1b1", device_map="auto", dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("data/prompt_data/final_peft")
    model = PeftModel.from_pretrained(base, "data/prompt_data/final_peft").eval()

    messages = [
        {"role": "system", "content": "You are a system prompt generator."},
        {"role": "user",   "content": "Create a system prompt for a travel advisor."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs, max_new_tokens=400, do_sample=True,
        temperature=0.4, top_p=0.85, repetition_penalty=1.4,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

---

## 📜 License

MIT License — free to use, modify, and distribute.