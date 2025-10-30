import urllib
import time
from pathlib import Path
import math
from datasets import (download_and_unzip_spam_data, 
                      create_balanced_dataset, 
                      random_split, SpamDataset)
import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

from model import GPTModel, load_model_config, customise_gpt
from utils.gpt_load import load_weights_into_gpt
from utils.gpt_download import download_and_load_gpt2
from utils.training_utils import train_classifier_simple, plot_values, calc_accuracy_loader


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
        
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###### Preparing Data #######

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "./data/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    print("\nDownloading Spam-Ham SMS data...")
    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    print("Dataset calss counts: ", balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # Test size is implied to be 0.2 as the remainder

    print("\nSaving data...")
    train_df.to_csv("./data/train.csv", index=None)
    validation_df.to_csv("./data/validation.csv", index=None)
    test_df.to_csv("./data/test.csv", index=None)


    ####### Creating Data loader ##########

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file="./data/train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file="./data/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file="./data/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    print(f"""\nTotal Batches
          {len(train_loader)} training batches
          {len(val_loader)} validation batches
          {len(test_loader)} test batches""")

      
    ###### Initializing Model and Training #########
    
    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG = load_model_config(CHOOSE_MODEL)
    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, 
                        models_dir="/content/drive/MyDrive/llm-from-scratch/work/gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    classifier = customise_gpt(model)

    print("Freezing pretrained weights...")
    for param in classifier.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Total trainable parameters after freezing: {total_params:,}")

    replace_linear_with_lora(classifier, rank=16, alpha=16)

    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Total trainable LoRA parameters: {total_params:,}")

    classifier.to(device)

    start_time = time.time()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    print("Training starts...")
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        classifier, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    ##### Saving model #######
    torch.save(classifier.state_dict(), "gpt2_classifier_lora_tuned.pth")

if __name__ == "__main__":
    main()