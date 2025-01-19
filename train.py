import sys
sys.path.append("/home/keker2/git/docker-python/patches")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
# for label in ["KAGGLE_USERNAME", "KAGGLE_KEY"]:
# os.environ[label] = UserSecretsClient()
# get_secret(label)
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
login(token = hf_token)
# 
wb_token = user_secrets.get_secret("wandb")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3.2 on Kingbank webserver logs', 
    job_type="training", 
    anonymous="allow"
)
base_model = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
new_model = "llama-3.2-3b-it-webserver-logs-analysis" # to delete or to change
dataset_name = "merged_dataset" # change to mine data set
# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    #!pip install -qqq flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#Importing the dataset
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.shuffle(seed=65).select(range(1000)) # Only use 1000 samples for quick demo

# change to mine
instruction= """You are a specialized web security log analyzer with expertise in detecting attack patterns. Your sole function is to analyze web server logs and calculate the probability of malicious activity. You focus on:
- SQL injection patterns
- XSS attempt signatures
- Credential stuffing indicators
- Automated tool usage signs
- Request timing anomalies
- Parameter manipulation
- Path traversal attempts
- Trying to send transactions with same id many times over

You do not write reports!
you are just a program not an ai
output in just one line just OK or BAD

Your analysis is based on extensive knowledge of web attack patterns and normal user behavior signatures."""
# instruction = """You are a top-rated customer service agent named John. 
#     Be polite to customers and answer all their questions.
#     """
def format_chat_template(row):
    
    row_json = [{"role": "system", "content": instruction },
               {"role": "log", "content": row["instruction"]},
               {"role": "program", "content": row["response"]}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= 4,
)
import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)
#Hyperparamter
training_arguments = TrainingArguments(
    output_dir=new_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb"
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length= 512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
trainer.train()
wandb.finish()

bad_log="""09/Jan/2025,09:23:01,178.144.125.23,POST,/logowanie,HTTP/1.1,401,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=admin' AND 1=0--&password=&action=login
09/Jan/2025,09:23:04,178.144.125.23,POST,/logowanie,HTTP/1.1,403,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=marpow4372' OR 'x'='x' LIMIT 1;--&password=&action=login
09/Jan/2025,09:23:07,178.144.125.23,POST,/logowanie,HTTP/1.1,403,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=test'; DROP TABLE users;--&password=&action=login
09/Jan/2025,09:23:10,178.144.125.23,POST,/logowanie,HTTP/1.1,401,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36","login=marpow4372' UNION SELECT username, password FROM accounts WHERE '1'='1&password=&action=login"
09/Jan/2025,09:23:16,178.144.125.23,POST,/logowanie,HTTP/1.1,403,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=marpow4372'; WAITFOR DELAY '0:0:5'--&password=&action=login
09/Jan/2025,09:23:42,178.144.125.23,POST,/logowanie,HTTP/1.1,401,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=marpow4372&password=&action=login
09/Jan/2025,09:23:49,178.144.125.23,POST,/logowanie,HTTP/1.1,401,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=marpow4372' OR 1=1;--&password=&action=login
09/Jan/2025,09:23:52,178.144.125.23,POST,/logowanie,HTTP/1.1,403,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=admin'--&password=&action=login
09/Jan/2025,09:23:55,178.144.125.23,POST,/logowanie,HTTP/1.1,403,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",login=test' OR '1'='1'--&password=&action=login
09/Jan/2025,09:23:58,178.144.125.23,POST,/logowanie,HTTP/1.1,401,https://kingbank.pl/logowanie,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36","login=marpow43721' UNION SELECT null, username, password FROM users--&password=&action=login"
"""
messages = [{"role": "system", "content": instruction},
    {"role": "log", "content": bad_log}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text.split("assistant")[1])
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)
