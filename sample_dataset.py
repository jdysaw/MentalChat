"""
从 SoulChatCorpus 数据集中采样子集，加速训练
"""
import json
import random
import os

def sample_dataset(input_path, output_path, num_samples=5000, seed=42):
    print(f"加载原始数据集: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"原始数据量: {len(data)} 条")
    random.seed(seed)
    sampled = random.sample(data, min(num_samples, len(data)))
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"采样完成: {len(sampled)} 条 -> {output_path}")
    return len(sampled)

if __name__ == '__main__':
    input_file = 'dataset/SoulChatCorpus/SoulChatCorpus-sft-multi-Turn.json'
    print("生成 5,000 条数据子集...")
    sample_dataset(input_file, 'dataset/soulchat_5k.json', 5000)
    print("\n生成 10,000 条数据子集...")
    sample_dataset(input_file, 'dataset/soulchat_10k.json', 10000)
    print("\n完成!")
