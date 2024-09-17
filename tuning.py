from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenized_dataset = load_from_disk('tokenized_llama_medical_dialogue')

training_args = TrainingArguments(
    output_dir="./llama_finetuned",      
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,       
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',                
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],   
    eval_dataset=tokenized_dataset["validation"]  
)

trainer.train()

trainer.save_model("./llama_finetuned")