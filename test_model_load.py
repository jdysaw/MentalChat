
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 50)
print("测试模型加载")
print("=" * 50)

model_path = "D:/models/Qwen1.5-1.8B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

try:
    print("\n[1/4] 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("[OK] Tokenizer加载成功")

    print("\n[2/4] 加载模型...")
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    print("[OK] 模型加载成功")

    print(f"\n[3/4] 模型已加载到: {model.device}")
    if device == "cuda":
        print(f"[4/4] 显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("\n" + "=" * 50)
    print("测试推理...")
    test_input = tokenizer("你好，请介绍一下自己", return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**test_input, max_new_tokens=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("[OK] 推理成功")
    print(f"回复: {response}")

except Exception as e:
    print(f"\n[ERROR] 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("所有测试通过！")

