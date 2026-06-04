"""
自动测试心理健康助手对话效果
模拟用户输入，测试模型回复质量
"""
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置路径
base_model_path = 'D:/models/Qwen1.5-1.8B-Chat'
lora_path = os.path.join(os.path.dirname(__file__), 'qwen_lora_5k')

print("=" * 60)
print("加载心理健康助手模型...")
print("=" * 60)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 加载 LoRA 权重
print(f"加载 LoRA 权重: {lora_path}")
model = PeftModel.from_pretrained(model, lora_path)
model.eval()
print("模型加载完成!\n")

def generate_response(prompt, max_new_tokens=256):
    """生成回复"""
    # 使用与 Web 界面相同的 system prompt
    system_prompt = "你是一个普通朋友，用日常自然的方式聊天。不要假设对方有心理问题，不要过度共情。像正常人一样聊天就好。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.85,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# 测试对话场景
test_scenarios = [
    # 场景1: 日常问候
    {
        "user": "你好",
        "expected": "正常问候回复，不过度共情"
    },
    # 场景2: 表达焦虑
    {
        "user": "我最近总是感到很焦虑，晚上睡不着觉",
        "expected": "提供具体的缓解焦虑建议"
    },
    # 场景3: 工作压力
    {
        "user": "工作压力太大了，经常加班到很晚",
        "expected": "理解并给出工作压力管理建议"
    },
    # 场景4: 情绪低落
    {
        "user": "心情不好，感觉对什么都没兴趣",
        "expected": "提供情绪调节建议"
    },
    # 场景5: 人际关系
    {
        "user": "我和同事关系不太好，有点困扰",
        "expected": "给出人际交往建议"
    },
    # 场景6: 简单闲聊
    {
        "user": "今天天气不错",
        "expected": "正常闲聊回复"
    }
]

print("=" * 60)
print("开始测试心理健康助手")
print("=" * 60)

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'='*60}")
    print(f"场景 {i}: {scenario['expected']}")
    print(f"{'='*60}")
    print(f"\n用户输入: {scenario['user']}")
    print("-" * 60)
    
    response = generate_response(scenario['user'])
    print(f"助手回复: {response}")
    print()

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
