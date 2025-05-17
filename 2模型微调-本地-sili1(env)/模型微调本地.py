from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import shutil
from traincallback import EvaluateCallback
# 加载数据集
dataset = load_dataset('json',data_files=r"C:\Users\hhhxj\Desktop\专利数据\3知识元标注\converted_data_formatted.json",split='train')

train_test_dataset = dataset.train_test_split(test_size=0.2)

device ='cpu'
model_name = 'deepseek-r1-distill-qwen-1.5b'
model_path =r"D:\deepseek微调\deepseek-1.5b"
output_dir = r"D:\deepseek微调\deepseek-1.5b-finetuned"

# 加载model
print('Loading tokenizer and model...')
tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          truse_reomote_code=True,
                                          padding_side='right',)
# 这是一个类用于加载因果模型的
model= AutoModelForCausalLM.from_pretrained(model_path)

input_text=dataset[0]['messages']

# print(tokenizer.special_tokens_map)
chat_template_text=tokenizer.apply_chat_template(
    input_text,
    tokenize=False,
    add_generation_prompt=False,
)
# print(chat_template_text)
# print(type(chat_template_text))

input_ids =tokenizer.encode(chat_template_text, return_tensors='pt')
# print(input_ids)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM"
)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Removed the existing output directory at {output_dir}")

os.makedirs(output_dir)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=train_test_dataset['test'],
    args=SFTConfig(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=10,
    ),
    peft_config=peft_config,
    callbacks=[EvaluateCallback(train_test_dataset['test'], tokenizer,model,train_test_dataset)]
)

trainable_params=0
all_params=0
for _,param in model.named_parameters():
    all_params+=param.numel()
    if param.requires_grad:
        trainable_params+=param.numel()

print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {all_params:,}")
print(f"Percentage of parameters being training: {100*trainable_params/all_params:.2f}%")

train_output = trainer.train()