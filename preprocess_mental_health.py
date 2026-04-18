"""
心理健康数据集预处理脚本
将原始心理健康对话数据转换为 MiniMind 训练格式
支持多种数据源格式转换
"""
import json
import os
from typing import List, Dict

def convert_efqa_to_minimind_format(
    input_file: str = None,
    output_file: str = 'dataset/mental_health_efqa.jsonl',
    max_length: int = 512
):
    """
    将 efaqa-corpus-zh 格式转换为 MiniMind SFT 格式
    
    输入格式：
    {
        "title": "咨询标题",
        "chats": [
            {"sender": "owner", "value": "用户消息"},
            {"sender": "audience", "value": "咨询师回复"}
        ]
    }
    
    输出格式：
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    converted_data = []
    skipped = 0
    
    # 如果提供了输入文件，从文件读取
    if input_file and os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    converted = process_efqa_item(item, max_length)
                    if converted:
                        converted_data.append(converted)
                except:
                    skipped += 1
                    continue
    else:
        print(f"未找到输入文件: {input_file}")
        print("使用内置示例数据进行演示")
        return False
    
    # 保存转换后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n数据转换完成！")
    print(f"成功转换: {len(converted_data)} 条")
    print(f"跳过: {skipped} 条")
    print(f"输出文件: {output_file}")
    
    return True

def process_efqa_item(item: Dict, max_length: int = 512) -> Dict:
    """处理单条 efaqa 数据"""
    try:
        chats = item.get('chats', [])
        if len(chats) < 2:
            return None
        
        # 将聊天记录转换为对话格式
        conversations = []
        for chat in chats:
            sender = chat.get('sender', '')
            content = chat.get('value', '')
            
            if len(content) > max_length:
                content = content[:max_length]
            
            if sender == 'owner':
                conversations.append({"role": "user", "content": content})
            elif sender == 'audience':
                conversations.append({"role": "assistant", "content": content})
        
        # 确保对话以用户开始，助手结束
        if len(conversations) >= 2:
            if conversations[0]["role"] == "assistant":
                conversations = conversations[1:]
            
            if conversations[-1]["role"] == "user":
                conversations = conversations[:-1]
            
            # 确保是完整的问答对
            if len(conversations) >= 2:
                return {"conversations": conversations}
        
        return None
    except:
        return None

def merge_datasets(
    dataset_files: List[str],
    output_file: str = 'dataset/mental_health_merged.jsonl'
):
    """
    合并多个心理健康数据集
    
    Args:
        dataset_files: 数据集文件列表
        output_file: 输出文件路径
    """
    merged_data = []
    
    for file_path in dataset_files:
        if not os.path.exists(file_path):
            print(f"警告：文件不存在 - {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'conversations' in item and len(item['conversations']) >= 2:
                        merged_data.append(item)
                except:
                    continue
    
    # 保存合并后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"合并完成！共 {len(merged_data)} 条数据")
    print(f"输出文件: {output_file}")
    
    return len(merged_data)

def split_dataset(
    input_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    划分训练集、验证集、测试集
    
    Args:
        input_file: 输入文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    import random
    
    # 读取所有数据
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                all_data.append(item)
            except:
                continue
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 划分数据集
    train_size = int(len(all_data) * train_ratio)
    val_size = int(len(all_data) * val_ratio)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # 保存数据集
    base_name = os.path.splitext(input_file)[0]
    train_file = f"{base_name}_train.jsonl"
    val_file = f"{base_name}_val.jsonl"
    test_file = f"{base_name}_test.jsonl"
    
    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(test_data, test_file)
    
    print(f"\n数据集划分完成！")
    print(f"训练集: {len(train_data)} 条 ({train_ratio*100}%)")
    print(f"验证集: {len(val_data)} 条 ({val_ratio*100}%)")
    print(f"测试集: {len(test_data)} 条 ({(1-train_ratio-val_ratio)*100}%)")

def save_jsonl(data: List[Dict], output_file: str):
    """保存为 JSONL 格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def dataset_statistics(input_file: str):
    """
    数据集统计分析
    
    Args:
        input_file: 输入文件路径
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except:
                continue
    
    print(f"\n=== 数据集统计 ===")
    print(f"总数据量: {len(data)} 条")
    
    if len(data) == 0:
        return
    
    # 统计对话轮数
    conv_lengths = []
    user_lengths = []
    assistant_lengths = []
    
    for item in data:
        conversations = item.get('conversations', [])
        conv_lengths.append(len(conversations))
        
        for conv in conversations:
            content_len = len(conv.get('content', ''))
            if conv['role'] == 'user':
                user_lengths.append(content_len)
            else:
                assistant_lengths.append(content_len)
    
    print(f"\n对话统计:")
    print(f"  平均对话轮数: {sum(conv_lengths)/len(conv_lengths):.2f}")
    print(f"  最大对话轮数: {max(conv_lengths)}")
    print(f"  最小对话轮数: {min(conv_lengths)}")
    
    print(f"\n内容长度统计:")
    print(f"  用户输入 - 平均: {sum(user_lengths)/len(user_lengths):.0f} 字符")
    print(f"  助手回复 - 平均: {sum(assistant_lengths)/len(assistant_lengths):.0f} 字符")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='心理健康数据集预处理')
    parser.add_argument('--action', type=str, default='stats',
                        choices=['convert', 'merge', 'split', 'stats'],
                        help='执行的操作')
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    if args.action == 'convert':
        convert_efqa_to_minimind_format(
            input_file=args.input,
            output_file=args.output or 'dataset/mental_health_efqa.jsonl'
        )
    elif args.action == 'merge':
        # 示例：合并多个数据集
        datasets = [
            'dataset/mental_health_sft.jsonl',
            'dataset/mental_health_efqa.jsonl'
        ]
        merge_datasets(
            dataset_files=datasets,
            output_file=args.output or 'dataset/mental_health_merged.jsonl'
        )
    elif args.action == 'split':
        input_file = args.input or 'dataset/mental_health_merged.jsonl'
        split_dataset(input_file)
    elif args.action == 'stats':
        input_file = args.input or 'dataset/mental_health_sft.jsonl'
        dataset_statistics(input_file)
