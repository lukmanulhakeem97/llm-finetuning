import torch
import tiktoken
from model import GPTModel
from utils.training import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    )


MODEL_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

def extract_response(response_text, input_text):
    response = response_text[len(input_text):].replace("### Response:", "").strip()
    response_list = response.split()
    if "Response:" in response_list:
        break_point_idx = response_list.index("Response:")
        response = " ".join(response_list[break_point_idx+1:])
    return response

def invoke(input_text, llm) -> str:

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {input_text}
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
            model=llm,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=256,
            context_size=MODEL_CONFIG["context_length"],
            eos_id=50256
        )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(generated_text, prompt)

    return response


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = "gpt2_instruct_tuned.pth"

    model = GPTModel(MODEL_CONFIG)
    ckpt = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # total_params = sum(p.numel() for p in model.parameters())
    # total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    # print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
    # print(f"Total number of parameters: {total_params:,}")

    # total_size_bytes_ = total_params * 4
    # total_size_mb_ = total_size_bytes_ / (1024 * 1024)
    # print(f"Total size of the model: {total_size_mb_:.2f} MB")

    print("This is an task solving chat llm. Give task as an instruction sentence.\n")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            else:
                model_response = invoke(user_input, model)
            print("Bot: ", model_response)
        except Exception as e:
            print("An Exception occured: ",e)
