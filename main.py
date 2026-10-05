from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk


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


if __name__ == "__main__":
    # model_name = "bigscience/bloomz-560m"
    model_name = "bigscience/bloomz-1b1"
    NUM_VIRTUAL_TOKEN = 4
    NUM_EPOCHS = 10

    # 1. base model
    # ---------------------------------------------------------
    base_model, base_tokenizer = load_base_model_and_tokenizer(model_name)

    # prompt = "there are two nice things that should matter to you: "
    # print(zero_shot(base_model, base_tokenizer, prompt))

    # 2. dataset
    # ---------------------------------------------------------
    dataset1_name = "fka/prompts.chat"
    dataset2_name = "Abirate/english_quotes"

    dataset1 = load_dataset(dataset1_name)
    print(dataset1.column_names)

    dataset2 = load_dataset(dataset2_name)
    print(dataset2.column_names)
