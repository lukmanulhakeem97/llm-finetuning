# Instruction Finetuning of GPT2
Supervised Finetuning of GPT2 pretrained model on custom instruction [dataset](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json). We use OpenAI gpt2-medium (355M) pretrained tensorflow weights after converting to pytorch format. Model architecture configuration is given `model_info.txt`.

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
**Ask Model:**
- Download pretrained `gpt2_instruct_tuned.pth` from my [huggingfaceHub](https://huggingface.co/lukmanulhakeem/gpt2-instruction-tuned/tree/main) and place it on cloned `llm-finetuning\gpt2-instruction-tuning` path.
- Run `inference.py`: Shows User-Bot interaction mode. You should ask model to solve the task as an instruction sentence, eg: `"Classify the following numbers as prime or composite: 11, 14, 19."`

**Finetuning:**
- Run `uv run finetune.py`, will generate `gpt2_instruct_tuned.pth`.

**Evaluation:**
- Finetuned model is evaluated with [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) method. It uses another LLM to evaluate the response of our model. We will be using test data partioned from the dataset `data/instruction-data.json` after running `finetune.py`.
- We use Ollama and `qwen2.5:7b` llm to run evaluation. Model can be any other regarding computaion ability of our machine.
     > Refer [ollama-setup](https://github.com/ollama/ollama) to run ollama and local model on your machine.
     > After installing ollama and llm, run ollama either via selecting ollama application or by running `ollama serve`.
- Run `evaluation.py` to evaluate the model.
- Average score after evaluation is 49.54 out of 100 on 110 test samples.

## Credits
- https://github.com/rasbt/LLMs-from-scratch
- https://amzn.to/4fqvn0D







