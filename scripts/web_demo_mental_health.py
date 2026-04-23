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
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

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

@st.cache_resource
def load_model():
    """加载Qwen-1.8B心理健康模型"""
    import locale
    if hasattr(locale, 'setlocale'):
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Qwen模型路径
    model_path = 'D:/models/Qwen1.5-1.8B-Chat'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None, None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[OK] Tokenizer loaded")
    except Exception as e:
        print(f"[ERROR] Tokenizer failed: {e}")
        return None, None, None
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        print(f"[OK] Qwen-1.8B model loaded")
    except Exception as e:
        print(f"[ERROR] Model failed: {e}")
        return None, None, None
    
    # 测试推理
    try:
        test_input = tokenizer("你好", return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**test_input, max_new_tokens=10)
        print(f"[OK] Inference test passed")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return None, None, None
    
    return model, tokenizer, device


def generate_response(model, tokenizer, device, user_input, history=None, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """使用Qwen模型生成心理健康回复"""
    if history is None:
        history = []
    
    conversation = history.copy()
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
        return "抱歉，我无法处理这个输入。请尝试重新提问。"
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            generated_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response if response.strip() else "抱歉，我无法生成回复。请尝试换个方式提问。"
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return "抱歉，生成回复时出错。请尝试重新提问。"


def main():
    st.set_page_config(
        page_title="心理健康助手",
        page_icon="🧠",
        layout="wide"
    )
    
    # 初始化聊天状态（必须在侧边栏之前初始化）
    init_chat_state()
    
    # 初始化历史对话状态
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # 加载模型
    with st.spinner("正在加载模型..."):
        try:
            model, tokenizer, device = load_model()
            st.success(f"✓ 模型加载成功 (设备: {device})")
        except Exception as e:
            st.error(f"✗ 模型加载失败: {e}")
            st.info("将使用模拟回复模式")
            model, tokenizer, device = None, None, None
    
    # 标题
    st.title("🧠 心理健康助手")
    st.caption("基于 MiniMind 超轻量语言模型 | 本地化部署 | 隐私保护")
    
    # 侧边栏 - 危机干预信息
    with st.sidebar:
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
                    # 使用真实模型
                    response = generate_response(
                        model, tokenizer, device,
                        prompt,
                        history=st.session_state.history,
                        max_new_tokens=200,  # 减少输出长度，避免生成过长无意义内容
                        temperature=0.7,     # 降低温度，提高稳定性
                        top_p=0.9            # 提高top_p，增加多样性
                    )
                    # 更新历史
                    st.session_state.history.append({"role": "user", "content": prompt})
                    st.session_state.history.append({"role": "assistant", "content": response})
                else:
                    # 使用模拟回复
                    response = simulate_mental_health_response(prompt)
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

def simulate_mental_health_response(user_input):
    """
    模拟心理健康助手回复
    实际使用时应替换为 MiniMind 模型的真实调用
    """
    # 这里只是一个示例，实际应该调用训练好的模型
    responses = {
        "焦虑": "我理解你的焦虑情绪。焦虑是很常见的心理状态。你可以尝试深呼吸练习：吸气4秒，屏住4秒，呼气6秒，重复几次。同时，试着把让你焦虑的事情写下来，分析哪些是可控的，哪些不可控。",
        "压力": "感到压力是很正常的。我建议你：1）制定合理的计划，把任务分解成小目标；2）保证充足的休息；3）适当运动帮助释放压力。你觉得哪种方法最适合你现在的状况？",
        "抑郁": "我听到你正在经历困难的时期。持续的情绪低落需要关注。建议你：1）保持规律作息；2）适当运动；3）和信任的人倾诉。如果这种情况持续两周以上，建议寻求专业心理咨询师的帮助。",
        "失眠": "睡眠问题确实很困扰人。你可以尝试：1）建立规律的作息时间；2）睡前1小时避免使用电子设备；3）睡前做些放松活动，如热水澡或轻音乐。",
        "孤独": "孤独感是很正常的情感体验。建立联系需要时间，你可以：1）参加感兴趣的社团或活动；2）主动和同学交流；3）做一个好的倾听者。慢慢来，友谊是慢慢建立的。",
    }
    
    # 简单的关键词匹配
    for keyword, response in responses.items():
        if keyword in user_input:
            return response
    
    # 默认回复
    return "感谢你的分享。我在这里倾听你，并尽力提供支持。你能再多告诉我一些你的感受吗？了解更多信息可以帮助我更好地理解你的情况。"

if __name__ == "__main__":
    main()
