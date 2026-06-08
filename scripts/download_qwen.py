import os
from huggingface_hub import snapshot_download

# 使用国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    # 设定下载目录为项目根目录下的 models 文件夹
    target_dir = r"f:\智能应用开发实践\MentalChat\models\Qwen1.5-1.8B-Chat"
    os.makedirs(target_dir, exist_ok=True)
    
    print("开始从 hf-mirror 下载 Qwen1.5-1.8B-Chat...")
    # 只下载必要文件，忽略一些大型的不必要文件（如果有的话）
    local_dir = snapshot_download(
        repo_id="Qwen/Qwen1.5-1.8B-Chat",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"\n[OK] 模型已成功下载并保存至: {local_dir}")

if __name__ == "__main__":
    main()
