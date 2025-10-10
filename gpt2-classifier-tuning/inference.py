import argparse
import torch
import tiktoken

from model import GPTModel, load_model_config, customise_gpt

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = load_model_config(CHOOSE_MODEL)

def classify(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 classifier Inference")
    parser.add_argument("user_input", type=str, help="enter message/email.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    checkpoint = torch.load("gpt2_classifier_tuned.pth", map_location=device, weights_only=True)

    base_model = GPTModel(BASE_CONFIG)
    classifier = customise_gpt(base_model)

    classifier.load_state_dict(checkpoint)
    classifier.to(device)
    classifier.eval()

    max_length = 120 # maximum train data sequence length
    pred_label = classify(
        args.user_input, classifier, tokenizer, device, max_length=max_length
    )

    print(f"Given text message predicted as {pred_label}")