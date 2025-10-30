import argparse
import torch
import tiktoken
import math

from model import GPTModel, load_model_config, customise_gpt

CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG = load_model_config(CHOOSE_MODEL)

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)

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

    checkpoint = torch.load("gpt2_classifier_lora_tuned.pth", map_location=device, weights_only=True)

    base_model = GPTModel(BASE_CONFIG)
    classifier = customise_gpt(base_model)

    replace_linear_with_lora(classifier, rank=16, alpha=16)

    classifier.load_state_dict(checkpoint)
    classifier.to(device)
    classifier.eval()

    max_length = 120 # maximum train data sequence length
    pred_label = classify(
        args.user_input, classifier, tokenizer, device, max_length=max_length
    )

    print(f"Given text message predicted as {pred_label}")