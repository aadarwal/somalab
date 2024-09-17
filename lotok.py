from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

token = "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"  
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=token)

dataset = load_dataset('json', data_files='path_to_your/medical_dialogue_dataset.json')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True, batch_size=4)

tokenized_dataset.save_to_disk('tokenized_llama_medical_dialogue')

print()