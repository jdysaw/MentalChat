"""
Qwen-1.8B LoRA 微调脚本
使用 SoulChatCorpus 心理健康数据集进行参数高效微调
"""
import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import warnings
warnings.filterwarnings('ignore')


def load_soulchat_dataset(data_path):
    """加载 SoulChatCorpus 数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        messages = item.get('messages', [])
        if len(messages) >= 2:
            examples.append({"messages": messages})
    
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='D:/models/Qwen1.5-1.8B-Chat')
    parser.add_argument('--sft_model_path', type=str, default=None, help='SFT微调后的模型路径（可选）')
    parser.add_argument('--data_path', type=str, default='../dataset/SoulChatCorpus/SoulChatCorpus-sft-multi-Turn.json')
    parser.add_argument('--output_dir', type=str, default='qwen_lora_1.8b')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--save_steps', type=int, default=1000)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen-1.8B LoRA 微调")
    print("=" * 60)
    
    # 确定模型路径
    model_path = args.sft_model_path if args.sft_model_path else args.model_path
    print(f"基础模型: {model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    # 加载tokenizer
    print("\n加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # 配置LoRA (不再使用 prepare_model_for_kbit_training)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print("\n加载数据集...")
    examples = load_soulchat_dataset(args.data_path)
    print(f"数据集大小: {len(examples)} 条")
    
    # 预处理数据
    def preprocess(examples):
        texts = []
        for messages in examples['messages']:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_tensors=None
        )
        
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    dataset = Dataset.from_dict({'messages': [ex['messages'] for ex in examples]})
    dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"预处理完成，样本数: {len(dataset)}")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=True,
        gradient_checkpointing=False,  # 关闭 gradient_checkpointing 避免梯度问题
        remove_unused_columns=False,
        report_to='none',
        dataloader_num_workers=0,
        optim='adamw_torch'
    )
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("开始 LoRA 训练...")
    print("=" * 60)
    trainer.train()
    
    # 保存LoRA权重
    print("\n保存模型...")
    trainer.save_model(args.output_dir)
    
    print(f"\nLoRA 训练完成! 权重保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
