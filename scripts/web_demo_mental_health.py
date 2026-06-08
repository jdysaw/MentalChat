"""
心理健康助手 Web 界面
基于 Streamlit 构建，支持危机识别和干预功能
"""
import streamlit as st
import json
import os
import sys
import time
import torch
import warnings
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
from peft import PeftModel

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')

def load_config():
    """从 config.json 读取配置"""
    default_config = {
        "base_model_path": "D:/models/Qwen1.5-1.8B-Chat",
        "lora_path": "",
        "max_new_tokens": 200,
        "temperature": 0.5,
        "top_p": 0.85
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                default_config.update(saved)
        except Exception:
            pass
    return default_config

def save_config(config):
    """保存配置到 config.json"""
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# 危机干预关键词
CRISIS_KEYWORDS = ['自杀', '自残', '不想活', '活着没意思', '伤害自己', 
                   '结束生命', '跳楼', '割腕', '死掉', '不想活了']

# 危机干预热线
CRISIS_HOTLINES = {
    "全国心理援助热线": "400-161-9995",
    "北京心理危机研究与干预中心": "010-82951332",
    "生命热线": "400-821-1215",
    "希望24热线": "400-161-9995",
}

# 危机干预回复模板
CRISIS_RESPONSE = """
⚠️ **重要提示**

我注意到你可能正在经历非常困难的时刻。请务必联系专业人士获取帮助：

📞 **24小时危机干预热线：**
""" + '\n'.join([f"- {name}：{phone}" for name, phone in CRISIS_HOTLINES.items()]) + """

请记住：
- 你并不孤单
- 有人愿意帮助你
- 困难是可以克服的

如果你现在处于危险中，请立即拨打 110 或 120。
"""

def detect_crisis(text):
    """检测文本是否包含危机关键词"""
    crisis_keywords = [kw for kw in CRISIS_KEYWORDS if kw in text]
    return len(crisis_keywords) > 0, crisis_keywords

def init_chat_state():
    """初始化聊天状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "crisis_detected" not in st.session_state:
        st.session_state.crisis_detected = False

def display_crisis_banner():
    """显示危机干预横幅"""
    st.warning(
        """
        ⚠️ **检测到你可能处于危机状态**
        
        如果你现在有伤害自己的想法，请立即联系专业人士：
        - 📞 全国心理援助热线：**400-161-9995**
        - 📞 北京心理危机干预中心：**010-82951332**
        - 📞 紧急求助：**110** 或 **120**
        """
    )

@st.cache_resource(max_entries=1, ttl=3600, show_spinner=False)
def load_model_cached(_base_model_path, _lora_path):
    """加载Qwen-1.8B心理健康模型 + LoRA微调权重（若失败则尝试探测并降级到 Ollama）"""
    import locale
    import psutil
    if hasattr(locale, 'setlocale'):
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 如果 _base_model_path 不存在，尝试探测本地 Ollama
    if not _base_model_path or not os.path.exists(_base_model_path):
        print(f"[INFO] Base model path not found: {_base_model_path}")
        print("[INFO] Falling back to check Ollama service...")
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if "gemma4:e2b" in models or len(models) > 0:
                    target_model = "gemma4:e2b" if "gemma4:e2b" in models else models[0]
                    print(f"[OK] Ollama detected. Using model: {target_model}")
                    return "ollama", target_model, "localhost (Ollama API)"
        except Exception as e:
            print(f"[INFO] Ollama connection failed: {e}")
        return None, None, "not_found"

    available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
    if device == 'cpu' and available_mem_gb < 4:
        print(f"[WARNING] Available RAM only {available_mem_gb:.1f}GB")
        # 内存不足也尝试探测下 Ollama
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if len(models) > 0:
                    target_model = "gemma4:e2b" if "gemma4:e2b" in models else models[0]
                    return "ollama", target_model, "localhost (Ollama API)"
        except:
            pass
        return None, None, "low_memory"

    # 确定 LoRA 路径
    if _lora_path and os.path.exists(_lora_path):
        lora_path = _lora_path
    else:
        lora_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trainer', 'qwen_lora_5k')
    if not os.path.exists(lora_path):
        lora_path = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(_base_model_path, trust_remote_code=True)
        print(f"[OK] Tokenizer loaded")
    except Exception as e:
        print(f"[ERROR] Tokenizer failed: {e}")
        # 尝试探测 Ollama
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if len(models) > 0:
                    target_model = "gemma4:e2b" if "gemma4:e2b" in models else models[0]
                    return "ollama", target_model, "localhost (Ollama API)"
        except:
            pass
        return None, None, None

    try:
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["low_cpu_mem_usage"] = True
        else:
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(_base_model_path, **load_kwargs)

        if lora_path and os.path.exists(os.path.join(lora_path, 'adapter_model.safetensors')):
            print(f"[OK] Loading LoRA from: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)

        print(f"[OK] Qwen-1.8B model loaded on {device}")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        # 尝试探测 Ollama
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                if len(models) > 0:
                    target_model = "gemma4:e2b" if "gemma4:e2b" in models else models[0]
                    return "ollama", target_model, "localhost (Ollama API)"
        except:
            pass
        return None, None, None

    try:
        test_input = tokenizer("你好", return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**test_input, max_new_tokens=10)
        print(f"[OK] Inference test passed")
    except Exception as e:
        print(f"[ERROR] Inference: {e}")
        return None, None, None

    return model, tokenizer, device


def generate_response_stream(model, tokenizer, device, user_input, history=None, max_new_tokens=512):
    """使用Qwen模型或Ollama流式生成心理健康回复"""
    if history is None:
        history = []
    
    # 添加 system prompt 引导模型提供更实用的建议
    system_prompt = "你是一个充满同理心且专业的心理健康助手。回复要简短、温暖、真诚，并且尽可能提供有建设性的帮助，就像微信聊天一样自然。"
    
    # 处理 Ollama 模型的逻辑
    if model == "ollama":
        import requests
        import json
        url = "http://localhost:11434/api/chat"
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "model": tokenizer,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.6,
                "top_p": 0.85
            }
        }
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=60)
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
            else:
                yield f"Ollama 服务返回错误代码: {resp.status_code}"
        except Exception as e:
            yield f"连接 Ollama 失败，请检查服务是否运行正常: {e}"
        return

    # 处理传统 HuggingFace Qwen 模型的逻辑
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(history)
    conversation.append({"role": "user", "content": user_input})
    
    try:
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
    except Exception as e:
        print(f"[ERROR] Tokenization failed: {e}")
        yield "抱歉，我无法处理这个输入。请尝试重新提问。"
        return
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        top_p=0.85,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    
    try:
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        yield "抱歉，生成回复时出错。请尝试重新提问。"


def main():
    st.set_page_config(
        page_title="心理健康助手",
        page_icon="🧠",
        layout="wide"
    )

    # 初始化聊天状态
    init_chat_state()

    # 初始化历史对话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    # === 阶段1：检查模型加载状态 ===
    # 使用 session_state 追踪加载状态，避免每次交互都阻塞
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = "pending"  # pending / loading / done / failed

    if "model_objects" not in st.session_state:
        st.session_state.model_objects = (None, None, None)

    if "load_message" not in st.session_state:
        st.session_state.load_message = ""

    if "load_progress" not in st.session_state:
        st.session_state.load_progress = 0

    # 读取配置
    config = load_config()

    # === 阶段2：渲染侧边栏（独立于模型加载） ===
    with st.sidebar:
        st.header("⚙️ 模型配置")
        with st.expander("模型路径设置"):
            new_base_path = st.text_input("基础模型路径", value=config["base_model_path"])
            new_lora_path = st.text_input("LoRA 权重路径（留空使用默认）", value=config.get("lora_path", ""))
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 保存配置"):
                    config["base_model_path"] = new_base_path
                    config["lora_path"] = new_lora_path
                    save_config(config)
                    st.cache_resource.clear()
                    # 重置加载状态
                    st.session_state.model_loaded = "pending"
                    st.session_state.model_objects = (None, None, None)
                    st.success("配置已保存，刷新页面重新加载")
            with col2:
                if st.button("🔄 重新加载"):
                    st.cache_resource.clear()
                    st.session_state.model_loaded = "pending"
                    st.session_state.model_objects = (None, None, None)
                    st.rerun()

        st.markdown("---")
        st.header("🆘 危机干预资源")
        for name, phone in CRISIS_HOTLINES.items():
            st.markdown(f"**{name}**\n\n📞 {phone}")

        st.markdown("---")
        st.info("""
        ⚠️ **免责声明**

        本助手仅供参考，不能替代专业心理咨询。
        如遇紧急情况，请立即联系专业机构。
        """)

        st.markdown("---")
        st.header("📊 使用统计")
        st.metric("对话轮数", len(st.session_state.messages) // 2)
        # 显示当前模式
        model, tokenizer, device = st.session_state.model_objects
        if model is not None:
            st.success(f"✅ 模型模式 ({device})")
        else:
            st.warning("⚠️ 模拟模式（关键词回复）")

    # === 阶段3：模型加载（仅在首次进入时阻塞） ===
    if st.session_state.model_loaded == "pending":
        st.title("🧠 心理健康助手")
        st.caption("基于 Qwen-1.8B + LoRA 微调 | 本地化部署 | 隐私保护")

        # 检查是否已缓存（用于显示即时状态）
        status_placeholder = None
        try:
            # 先用状态显示加载进度
            status_placeholder = st.status("正在加载模型中...请稍候（约 10-20 秒）", expanded=True)
            progress_bar = status_placeholder.progress(0)
            msg_text = status_placeholder.empty()

            st.session_state.model_loaded = "loading"

            # 模拟进度（实际缓存在后台完成）
            phases = [
                (5, "检查环境..."),
                (15, "正在加载 Tokenizer..."),
                (35, f"正在加载模型..."),
                (70, "正在加载 LoRA 微调权重..."),
                (85, "进行推理测试..."),
                (95, "收尾中..."),
            ]
            for pct, msg in phases:
                progress_bar.progress(pct / 100)
                msg_text.markdown(f"**{msg}**")
                time.sleep(0.3)

            # 真正的模型加载（走缓存）
            model, tokenizer, device = load_model_cached(
                config["base_model_path"],
                config.get("lora_path", ""),
            )

            if model is not None:
                st.session_state.model_objects = (model, tokenizer, device)
                st.session_state.model_loaded = "done"
                progress_bar.progress(1.0)
                msg_text.markdown("**✅ 模型加载成功！**")
                status_placeholder.update(label="✅ 模型加载成功！", state="complete", expanded=False)
            elif device == "not_found":
                st.session_state.model_loaded = "failed"
                status_placeholder.update(label="⚠️ 未找到本地模型", state="error", expanded=False)
                status_placeholder.markdown("将使用模拟回复模式。可修改侧边栏的模型路径后刷新重试。")
            elif device == "low_memory":
                import psutil
                avail = psutil.virtual_memory().available / (1024 ** 3)
                st.session_state.model_loaded = "failed"
                status_placeholder.update(label=f"❌ 内存不足（可用 {avail:.1f}GB）", state="error", expanded=True)
                status_placeholder.markdown("请关闭其他程序释放内存后，点击侧边栏「🔄 重新加载」按钮重试。")
            else:
                st.session_state.model_loaded = "failed"
                status_placeholder.update(label="❌ 模型加载失败", state="error", expanded=False)
                status_placeholder.markdown("转为模拟回复模式。检查模型路径是否正确。")
        except Exception as e:
            st.session_state.model_loaded = "failed"
            st.error(f"加载异常: {e}")
            if status_placeholder:
                status_placeholder.update(label=f"❌ 异常: {e}", state="error")

        time.sleep(1)
        st.rerun()

    # === 阶段4：显示聊天界面 ===
    model, tokenizer, device = st.session_state.model_objects

    # 标题
    st.title("🧠 心理健康助手")
    st.caption("基于 Qwen-1.8B + LoRA 微调 | 本地化部署 | 隐私保护")
    
    # 显示危机干预横幅
    if st.session_state.crisis_detected:
        display_crisis_banner()
    
    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 聊天输入
    if prompt := st.chat_input("请输入你想说的话..."):
        # 危机检测
        is_crisis, keywords = detect_crisis(prompt)
        
        if is_crisis:
            st.session_state.crisis_detected = True
            st.warning(f"⚠️ 检测到危机关键词：{', '.join(keywords)}")
            
            # 直接显示危机干预信息
            with st.chat_message("assistant"):
                st.markdown(CRISIS_RESPONSE)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": CRISIS_RESPONSE})
            st.rerun()
        
        # 正常对话
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 使用模型生成回复
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                if model is not None:
                    # 使用真实模型，改为流式输出
                    response_gen = generate_response_stream(
                        model, tokenizer, device,
                        prompt,
                        history=st.session_state.history,
                        max_new_tokens=200
                    )
                    response = st.write_stream(response_gen)
                    # 更新历史
                    st.session_state.history.append({"role": "user", "content": prompt})
                    st.session_state.history.append({"role": "assistant", "content": response})
                else:
                    # 使用模拟回复
                    response = simulate_mental_health_response(prompt, history=st.session_state.history)
                    st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

def simulate_mental_health_response(user_input, history=None):
    """
    模拟心理健康助手回复（含上下文感知）
    实际使用时应替换为 MiniMind 模型的真实调用
    """
    import random

    # 扩展关键词库，分组匹配
    keyword_groups = {
        "焦虑": [
            "我理解你的焦虑情绪。焦虑是很常见的心理状态。你可以尝试深呼吸练习：吸气4秒，屏住4秒，呼气6秒，重复几次。试着把让你焦虑的事情写下来，分析哪些是可控的。",
            '面对焦虑时，试试「5-4-3-2-1」grounding技巧：说出5样看到的东西、4样摸到的、3样听到的、2样闻到的、1样尝到的。你感觉怎么样？',
            "焦虑往往来自于对未来的不确定。把注意力拉回当下，关注此刻你能做的事。有什么具体的事情在困扰你吗？",
        ],
        "压力": [
            "感到压力是很正常的。我建议：1）把任务分解成小目标；2）保证充足休息；3）适当运动释放压力。你觉得哪种方法适合你？",
            "压力大的时候，试试番茄工作法：专注25分钟，休息5分钟。记得给自己留出放松的时间。",
            "工作/学习压力大的时候，看看能不能调整期望值——不需要每件事都做到完美。你能说说什么让你最感压力吗？",
        ],
        "抑郁": [
            "持续的情绪低落需要关注。建议：1）保持规律作息；2）适当运动；3）和信任的人倾诉。如果持续两周以上，建议寻求专业帮助。",
            "情绪低落时，试着做一件小事——整理桌面、散步10分钟、听一首喜欢的歌。小行动有时能带来变化。",
            "你愿意和我多聊聊最近发生了什么吗？有时候说出来本身就是一种释放。",
        ],
        "失眠": [
            "睡眠问题确实困扰人。可以尝试：1）固定作息时间；2）睡前1小时远离电子设备；3）做些放松活动如热水澡或轻音乐。",
            '睡不好时，试试「478呼吸法」：吸气4秒，屏气7秒，呼气8秒，重复几次。这能帮身体放松下来。',
            "如果躺了20分钟还睡不着，建议起床做点轻松的事（如看书），等有困意再上床。不要在床上焦虑。",
        ],
        "孤独": [
            "孤独感是很正常的情感体验。可以试试：1）参加感兴趣的社团或活动；2）主动和同学交流；3）做一个好的倾听者。友谊是慢慢建立的。",
            "感到孤独的时候，给家人或老朋友发条消息吧。有时候主动迈出第一步，会发现对方也在等你联系。",
            "独处和孤独不一样——独处可以是一种享受。如果感到孤独，可以试着养个爱好，或者参与一些线上社群活动。",
        ],
        "你好": [
            "你好！今天过得怎么样？有什么想聊聊的吗？",
            "嗨！有什么我可以帮你的吗？或者随便聊聊也行～",
            "你好呀！最近怎么样？",
        ],
        "谢谢": [
            "不客气！有什么随时找我聊～",
            "没事的，我在这儿陪着呢。",
            "不用谢～希望你的心情好一点了！",
        ],
        "生气|愤怒|烦躁|烦": [
            "生气是正常的情绪，重要的是怎么处理它。试试深呼吸，给自己10秒钟冷静一下。",
            "听起来你今天遇到了一些烦心事。要不要说说发生了什么？有时候说出来气就消了一半。",
        ],
        "考试|学习|复习": [
            "学习压力大是正常的。试试制定一个合理的复习计划，劳逸结合。你现在的复习进度怎么样？",
            "考试期间记得照顾好自己：按时吃饭、保证睡眠、适当运动。效率比时间更重要。",
        ],
        "朋友|同学|室友": [
            "人际关系有时确实让人困扰。能具体说说发生了什么吗？我们可以一起想想办法。",
            "处理人际关系最重要的是沟通。试试坦诚地表达你的感受，同时也要倾听对方的想法。",
        ],
        "累|疲惫|没精神": [
            "累了就休息一下吧。身体是革命的本钱，适当的休息才能走更远的路。",
            "持续疲惫可能说明你需要调整一下生活节奏。有没有什么你可以暂时放下的任务？",
        ],
    }

    # 遍历关键词组进行匹配
    for keywords, responses in keyword_groups.items():
        for kw in keywords.split("|"):
            if kw in user_input:
                return random.choice(responses)

    # 多个不同的默认回复，增加多样性
    default_responses = [
        "嗯，我在听。你想说什么都可以，不需要顾虑。",
        "谢谢你的分享。能多说一点吗？我想更好地理解你的情况。",
        "我明白你的意思了。你对此有什么感觉？",
        "嗯，继续说说你的想法吧，我在这儿听着。",
        "听起来这对你来说挺重要的。能展开讲讲吗？",
        "我理解。有时候把话说出来，心里就会轻松一些。",
    ]
    return random.choice(default_responses)

if __name__ == "__main__":
    main()
