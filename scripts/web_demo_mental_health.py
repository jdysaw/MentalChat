"""
心理健康助手 Web 界面
基于 Streamlit 构建，支持危机识别和干预功能
"""
import streamlit as st
import json
import os
import sys
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def main():
    st.set_page_config(
        page_title="心理健康助手",
        page_icon="🧠",
        layout="wide"
    )
    
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
    
    # 初始化聊天状态
    init_chat_state()
    
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
        
        # TODO: 这里应该调用 MiniMind 模型生成回复
        # 由于当前环境没有 GPU，这里使用模拟回复
        with st.chat_message("assistant"):
            # 模拟模型回复（实际使用时替换为真实模型调用）
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
