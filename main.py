from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base_model_and_tokenizer(model_name):
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return original_model, tokenizer


if __name__ == "__main__":
    # model_name = "bigscience/bloomz-560m"
    model_name = "bigscience/bloomz-1b1"
    NUM_VIRTUAL_TOKEN = 4
    NUM_EPOCHS = 10

    base_model, base_tokenizer = load_base_model_and_tokenizer(model_name)
