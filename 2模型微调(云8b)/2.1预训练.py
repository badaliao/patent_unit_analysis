import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
from datasets import load_dataset

# 定义模型名称
model_name = "/root/autodl-tmp/deepseek-r1-8b"  # 模型文件夹路径

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'  # 确保 padding_side 为 'right'

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # device_map={"": 0},  # 将模型加载到第一个 GPU
    device_map="auto",  # 将模型加载到第一个 GPU
    trust_remote_code=True  # 确保加载自定义代码

)

# LoRA 配置
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # 微调模型为自回归模型
    r=8,  # 将LoRA秩调整为8
    lora_alpha=32,  # LoRA 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标模块，根据LLaMA3模型结构指定
    lora_dropout=0.05,  # Dropout 概率
    bias="none",  # 不训练 bias
    init_lora_weights=True,  # 初始化 LoRA 层权重
    inference_mode=False  # 允许训练
)

# 将LoRA配置应用到模型
model = get_peft_model(model, lora_config)

# 定义训练参数
training_arguments = TrainingArguments(
    output_dir="./Llama3_8b_LoRA_2025",
    eval_strategy="no",  # 禁用评估
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,  # 适当减小批次大小
    gradient_accumulation_steps=1,  # 设置为1以减少显存占用
    per_device_eval_batch_size=8,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=80,
    learning_rate=1e-4,
    fp16=True,  # 启用混合精度训练
    bf16=False,
    num_train_epochs=10,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
)

# 加载已经处理好的数据
dataset = load_dataset("json", data_files='./converted_data_formatted.json', split='train')

# 创建训练器
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()
