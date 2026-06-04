"""
测试 LoRA 微调后的心理健康模型效果
"""
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置路径
base_model_path = 'D:/models/Qwen1.5-1.8B-Chat'
lora_path = os.path.join(os.path.dirname(__file__), 'qwen_lora_5k')

print("=" * 60)
print("加载模型...")
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
print("模型加载完成!")

def generate_response(prompt, max_new_tokens=256):
    """生成心理健康回复"""
    messages = [
        {"role": "system", "content": "你是一个专业的心理健康咨询师，请用温暖、专业的语气回答用户的问题。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# 测试问题
test_questions = [
    "我最近总是感到很焦虑，晚上睡不着觉，怎么办？",
    "我工作压力很大，经常感到疲惫，有什么建议吗？",
    "我情绪很低落，对什么都提不起兴趣，是不是抑郁了？",
    "我和同事关系不好，总是被孤立，我该怎么办？"
]

print("\n" + "=" * 60)
print("测试模型效果")
print("=" * 60)

for i, question in enumerate(test_questions, 1):
    print(f"\n问题 {i}: {question}")
    print("-" * 60)
    response = generate_response(question)
    print(f"回答: {response}")
    print("=" * 60)

print("\n测试完成!")
