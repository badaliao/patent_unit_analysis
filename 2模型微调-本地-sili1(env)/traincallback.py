import os
import json
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import torch
import datetime

class EvaluateCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer,model,test_dataset):
        self.test_dataset=test_dataset
        self.tokenizer=tokenizer
        self.epoch=0
        self.model=model
        self.eval_dataset=eval_dataset
        self.eval_losses=[]
        self.train_losses=[]
        self.epoch=[]

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nEvaluating model after epoch...{self.epoch} ")
        #存储当前的模型损失
        if state.log_history:
            latest_loss = state.log_history[-1].get("loss")
            if latest_loss is not None:
                self.eval_losses.append(latest_loss)
                self.epoch.append(self.epoch)

        self.model.eval()
        total_eval_loss = 0
        num_eval_samples = 0

        with torch.no_grad():
            for i in range(min(1,len(self.test_dataset))):
                user_input = self.test_dataset[i]["messages"][1]["content"]
                system_prompt =self.test_dataset[i]["messages"][0]["content"]

            #准备输入数据
            messages=[
                {"role":"system","content":system_prompt}
                ,{"role":"user","content":user_input}
            ]
            chat_template_text=self.tokenizer.apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=False)

            model_inputs = self.tokenizer(
                [chat_template_text],
                return_tensors="pt",
                ).to(self.model.device)

            #生成回复
            generated_ids =self.model.generate(
                **model_inputs,
                max_length=200,
                num_return_sequences=1,
                do_sample=False,
                tmperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                pad_token_type_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=False,
            )

            outputs=self.model(**model_inputs,labels=model_inputs.input_ids)
            loss = outputs.loss.item()
            total_eval_loss += loss
            num_eval_samples += 1

            generated_text = self.tokenizer.batch_decode(
                [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)],
                skip_special_tokens=True,
                )[0]

            print(f"\nTest sample {i+1}:")
            print(f"Input:{user_input}")
            print(f"Output:{generated_text}")
            print(f"Loss:{loss}")
            print("-"*50)

        avg_eval_loss = total_eval_loss / num_eval_samples if num_eval_samples > 0 else 0.0
        self.eval_losses.append(avg_eval_loss)

        metrics ={
        'epochs': self.epochs,
        'eval_losses': self.eval_losses,
        'train_losses': self.train_losses,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'current_epoch':self.epoch,
        }

        os.makedirs('losses', exist_ok=True)

        with open('losses/training_metrics.json', "w") as f:  # Fix: Corrected file path
            json.dump(metrics, f, indent=2)

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.trian_losses, 'b-',label="Training_loss")
        plt.plot(range(len(self.eval_losses)), self.eval_losses, 'r-',label="Evaluation_loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig("losses/Training_progress.png")
        plt.close()

        self.epoch +=1
        self.model.train()