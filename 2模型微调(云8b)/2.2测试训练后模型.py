from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载训练后的模型
model_path = "./Llama3_8b_LoRA_2025"  # 训练后的模型路径
model = AutoModelForCausalLM.from_pretrained(model_path)

# 加载训练后的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 确保模型在 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到 GPU 或 CPU

# 输入文本
input_text = "Your input text here"

# 编码输入文本并将输入移动到 GPU
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 使用模型生成输出
with torch.no_grad():  # 推理时不需要计算梯度
    outputs = model.generate(inputs['input_ids'], max_length=100)

# 解码生成的输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
