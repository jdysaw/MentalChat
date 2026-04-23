@echo off
echo ============================================================
echo Qwen-1.8B 心理健康模型微调
echo ============================================================
echo.
echo 请选择微调方式:
echo 1. SFT 全量微调 (需要约20GB显存，训练较慢)
echo 2. LoRA 微调 (推荐，约6GB显存，训练快)
echo.
set /p choice=请输入选项 (1 或 2): 

if "%choice%"=="1" goto sft
if "%choice%"=="2" goto lora
echo 无效选项，退出。
pause
exit

:sft
echo.
echo 开始 SFT 全量微调...
cd trainer
python train_qwen_sft.py --epochs 2 --batch_size 1 --grad_accum 8 --lr 2e-5 --max_length 1024 --save_steps 2000
pause
exit

:lora
echo.
echo 开始 LoRA 微调...
cd trainer
echo.
echo 是否使用 SFT 微调后的模型作为基础? (如果已执行过SFT)
echo 1. 是 (使用 SFT 模型)
echo 2. 否 (使用原始 Qwen 模型)
set /p use_sft=请输入选项:
if "%use_sft%"=="1" (
    python train_qwen_lora.py --sft_model_path qwen_sft_1.8b --epochs 3 --batch_size 2 --grad_accum 4 --lr 1e-4 --max_length 512 --save_steps 1000
) else (
    python train_qwen_lora.py --epochs 3 --batch_size 2 --grad_accum 4 --lr 1e-4 --max_length 512 --save_steps 1000
)
pause
exit
