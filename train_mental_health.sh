#!/bin/bash
# 心理健康助手 - MiniMind 训练脚本 (Linux/macOS)

echo "========================================"
echo "  心理健康助手 - MiniMind 训练脚本"
echo "========================================"
echo ""

# 步骤 1: 预训练
echo "[步骤 1/3] 预训练（学习基础知识）"
echo "----------------------------------------"
cd trainer
python train_pretrain.py --epochs 1 --batch_size 2 --max_seq_len 128 --save_interval 5
cd ..

echo ""
echo "[步骤 2/3] 监督微调（学习对话方式）"
echo "----------------------------------------"
cd trainer
python train_full_sft.py --epochs 1 --batch_size 2 --max_seq_len 128 --save_interval 5
cd ..

echo ""
echo "[步骤 3/3] 心理健康 LoRA 微调"
echo "----------------------------------------"
cd trainer
python train_lora.py \
    --data_path ../dataset/mental_health_sft.jsonl \
    --lora_name mental_health \
    --epochs 3 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --max_seq_len 128 \
    --save_interval 5
cd ..

echo ""
echo "========================================"
echo "  训练完成！"
echo "========================================"
echo ""
echo "测试命令: python eval_llm.py --weight full_sft --lora_weight mental_health"
echo "Web界面: streamlit run scripts/web_demo_mental_health.py"
echo ""
