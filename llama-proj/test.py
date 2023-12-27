from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import cuda
from peft import LoraConfig, get_peft_model
import torch

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

assert cuda.device_count() > 1



model_path = '/Data/ikhaja2/llama_proj/results_13b/checkpoint-100'

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

lora_config = LoraConfig.from_pretrained('outputs')

model = get_peft_model(model, lora_config)
#model = model.to(device)
model = torch.nn.DataParallel(model)
model.to('cuda')

with open('prompts.txt', 'r') as f:
    text = f.read()

questions = text.split('\n\n')

batch_size = 5  #batch by 5
batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]

with open('tuned-output.txt', 'w') as f:
    for batch in batches:
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=2048)
        for output in outputs:
            print(tokenizer.decode(output, skip_special_tokens=True))
            print('\n')
            print("-----------")
            print('\n')