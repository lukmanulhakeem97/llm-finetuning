# Classification Finetuning of GPT2
Finetuning ol GPT2 pretrained model on SPAM message classsification [dataset](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip). We use OpenAI gpt2 pretrained tensorflow weights after converting to pytorch format. Model architecture configuration is given `model_info.txt`.

## Setup
Pre-requisites are `python<=3.13` and `uv` package manger, instructions to set up can be found [here](https://docs.astral.sh/uv/getting-started/).
1. **Clone this repository**
   
   Either by download as zip option or by `git clone https://github.com/lukmanulhakeem97/llm-finetuning.git` command in CLI tool.
2. **Create an python environment and install dependencies**

   create environment: `uv venv [name]`, name is optional.
   
   Navigate to cloned repo directory and install dependency given in `pyproject.toml` file:
      > `cd llm-finetuning`,
      
      > `uv sync`.
4. Activate `venv` by `.\.venv\Scripts\activate`

## Run the code
**Predict Spam or not:**
- Download pretrained `gpt2_classifier_tuned.pth` from my [huggingfaceHub](https://huggingface.co/lukmanulhakeem/gpt2-classifier-tuned/tree/main) and place it on cloned `llm-finetuning\gpt2-classifier-tuning` path.
- Run `inference.py` with the input message: `uv run inference.py "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."`.

**Finetuning:**
- Run `uv run finetune.py`, will generate `gpt2_classifier_tuned.pth`.

## Credits
- https://github.com/rasbt/LLMs-from-scratch
- https://amzn.to/4fqvn0D







