import json

# Function to convert each line into valid JSON format
def convert_to_valid_json(lines):
    valid_json_lines = []
    for line in lines:
        try:
            # Replace single quotes with double quotes and load as JSON
            valid_line = json.loads(line.replace("'", '"'))
            valid_json_lines.append(valid_line)
        except json.JSONDecodeError as e:
            # If there's an error, add a note about the faulty line
            valid_json_lines.append(f"Error in line: {line} - {e}")
    return valid_json_lines

with open('jsonl_qa.txt') as f:
# Applying the conversion function to the content
    valid_json_content = convert_to_valid_json(f.read())

# Displaying the first few converted lines to verify the correction
print(valid_json_content[:5])

from peft import LoraConfig, get_peft_model
from datasets import Dataset

"""instatiating the model and quantizing config"""

model_name = '/Data/ikhaja2/llama_proj/results_13b/checkpoint-100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype= bfloat16
)

model_path = 'meta-llama/Llama-2-13b-chat-hf'

model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig.from_pretrained('outputs')

model = get_peft_model(model, lora_config)



tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token

#model.eval()

'''text = "What four main types of actions involve databases? Briefly discuss each."

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=2048)'''

from datasets import Dataset

train_dataset = Dataset.from_dict({
    'text': [item for item in dataset]
})



lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

from transformers import TrainingArguments

output_dir = "./qa_results_13b"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    num_train_epochs = 3
)

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,

)

trainer.train()

model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")