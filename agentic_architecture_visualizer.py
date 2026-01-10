# -*- coding: utf-8 -*-
"""
Agentic Architecture 可视化系统

使用Streamlit构建的交互式界面，用于展示和使用所有Agentic Architecture的示例。

功能特点：
- 支持所有Agentic Architecture示例的可视化展示
- 直观的架构选择界面
- 实时显示分析过程和日志
- 美观的结果展示
- 多智能体与单智能体系统对比
- 支持自定义参数配置

运行方式：
```bash
streamlit run agentic_architecture_visualizer.py
```
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from rich.console import Console

# 导入必要的库用于动态加载模块
import importlib.util
import sys

# 加载环境变量
load_dotenv()

# 初始化控制台
console = Console()

# 设置页面标题和布局
st.set_page_config(
    page_title="Agentic Architecture 可视化系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-bottom: 10px;
    }
    .analysis-section {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .agent-title {
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .monolithic-title {
        color: #FF9800;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .final-report {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .log-section {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 14px;
        white-space: pre-wrap;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏
st.sidebar.title("🤖 Agentic Architecture 可视化系统")
st.sidebar.markdown("---")

# 选择架构示例
st.sidebar.subheader("选择架构示例")
architecture_choices = [
    "01 - 反思型智能体 (Reflection)",
    "02 - 工具使用智能体 (Tool Use)",
    "03 - 反应型智能体 (ReAct)",
    "04 - 规划型智能体 (Planning)",
    "05 - 多智能体系统 (Multi-Agent)",
    "06 - 规划→执行→验证智能体 (Planner→Executor→Verifier)",
    "07 - 黑板系统 (Blackboard System)"
]
selected_architecture = st.sidebar.selectbox("", architecture_choices)

# 模型选择
st.sidebar.subheader("选择模型")
model_choices = {
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528"
}
selected_model = st.sidebar.selectbox("", list(model_choices.keys()))
# 设置环境变量
os.environ["MODELSCOPE_MODEL_ID"] = model_choices[selected_model]

# API密钥检查
api_key = os.environ.get("MODELSCOPE_API_KEY")
if not api_key:
    st.sidebar.error("未找到MODELSCOPE_API_KEY环境变量")
    st.sidebar.info("请创建.env文件并设置API密钥")
    api_key_input = st.sidebar.text_input("或直接输入API密钥", type="password")
    if api_key_input:
        os.environ["MODELSCOPE_API_KEY"] = api_key_input

# 主界面
st.title("📊 Agentic Architecture 可视化系统")

# 定义各个架构的可视化函数
def visualize_reflection():
    """可视化反思型智能体"""
    st.markdown("### 01 - 反思型智能体 (Reflection)")
    
    # 加载01_reflection模块
    spec = importlib.util.spec_from_file_location("reflection", "01_reflection.py")
    reflection = importlib.util.module_from_spec(spec)
    sys.modules["reflection"] = reflection
    spec.loader.exec_module(reflection)
    
    # 从模块中导入所需函数和类
    init_llm = reflection.init_llm
    build_app = reflection.build_app
    run_workflow = reflection.run_workflow
    print_before_after = reflection.print_before_after
    
    # 用户输入区域
    default_request = "Write a Python function to find the nth Fibonacci number."
    user_request = st.text_area("输入您的请求", value=default_request, height=100)
    
    # 执行按钮
    if st.button("开始执行反思工作流"):
        # 检查API密钥
        if not os.environ.get("MODELSCOPE_API_KEY"):
            st.error("请先设置API密钥")
        else:
            with st.spinner("正在初始化系统..."):
                # 初始化LLM
                llm = init_llm()
                
                # 构建工作流
                app = build_app(llm)
            
            st.success("系统初始化完成！")
            
            # 创建日志显示区域
            logs_container = st.empty()
            log_content = ""
            
            # 重定向控制台输出到日志区域
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # 执行工作流
                final_state = run_workflow(app, user_request)
                
            # 获取控制台输出
            log_content = f.getvalue()
            
            # 显示日志
            st.markdown("### 执行日志")
            st.text_area("", value=log_content, height=300, disabled=True)
            
            # 显示结果
            st.markdown("### 执行结果")
            
            # 显示初稿
            if "draft" in final_state:
                st.markdown("#### 1. 初稿")
                explanation = final_state["draft"].get("explanation", "")
                if explanation:
                    st.markdown(f"**说明**：{explanation}")
                code = final_state["draft"].get("code", "")
                if code:
                    st.code(code, language="python")
            
            # 显示评审结果
            if "critique" in final_state:
                st.markdown("#### 2. 评审")
                critique = final_state["critique"]
                st.json(critique)
            
            # 显示改写后的代码
            if "refined_code" in final_state:
                st.markdown("#### 3. 改写后")
                refined_code = final_state["refined_code"].get("refined_code", "")
                if refined_code:
                    st.code(refined_code, language="python")
                refinement_summary = final_state["refined_code"].get("refinement_summary", "")
                if refinement_summary:
                    st.markdown(f"**改进说明**：{refinement_summary}")


def visualize_tool_use():
    """可视化工具使用智能体"""
    st.markdown("### 02 - 工具使用智能体 (Tool Use)")
    
    # 加载02_tool_use模块
    spec = importlib.util.spec_from_file_location("tool_use", "02_tool_use.py")
    tool_use = importlib.util.module_from_spec(spec)
    sys.modules["tool_use"] = tool_use
    spec.loader.exec_module(tool_use)
    
    # 从模块中导入所需函数和类
    init_llm = tool_use.init_llm
    build_app = tool_use.build_app
    run_workflow = tool_use.run_workflow
    
    # 用户输入区域
    default_request = "请对这段话做简单文本管线：'LangGraph makes it easier to build stateful AI workflows.' 标准化、分词、提取5个关键词，最后结合当前时间渲染为 Markdown 报告。"
    user_request = st.text_area("输入您的请求", value=default_request, height=100)
    
    # 执行按钮
    if st.button("开始执行工具使用工作流"):
        # 检查API密钥
        if not os.environ.get("MODELSCOPE_API_KEY"):
            st.error("请先设置API密钥")
        else:
            with st.spinner("正在初始化系统..."):
                # 初始化LLM
                llm = init_llm()
                
                # 构建工作流
                app = build_app(llm)
            
            st.success("系统初始化完成！")
            
            # 创建日志显示区域
            logs_container = st.empty()
            log_content = ""
            
            # 重定向控制台输出到日志区域
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # 执行工作流
                final_state = run_workflow(app, user_request)
            
            # 获取控制台输出
            log_content = f.getvalue()
            
            # 显示日志
            st.markdown("### 执行日志")
            st.text_area("", value=log_content, height=300, disabled=True)
            
            # 显示结果
            st.markdown("### 执行结果")
            
            # 显示工具计划
            if "tool_plan" in final_state:
                st.markdown("#### 1. 工具计划")
                plan_summary = final_state["tool_plan"].get("plan_summary", "")
                if plan_summary:
                    st.markdown(f"**计划摘要**：{plan_summary}")
                planned_calls = final_state["tool_plan"].get("planned_calls", [])
                if planned_calls:
                    for i, call in enumerate(planned_calls):
                        st.markdown(f"**步骤 {i+1}**：{call.get('tool_name')}")
                        st.markdown(f"- 理由：{call.get('reason')}")
                        st.markdown(f"- 参数：{call.get('arguments')}")
                        if call.get('assign_to'):
                            st.markdown(f"- 结果保存到：{call.get('assign_to')}")
                        st.markdown("")
            
            # 显示工具执行结果
            if "tool_results" in final_state:
                st.markdown("#### 2. 执行结果")
                execution_summary = final_state["tool_results"].get("execution_summary", "")
                if execution_summary:
                    st.markdown(f"**执行摘要**：{execution_summary}")
                results = final_state["tool_results"].get("results", [])
                if results:
                    for result in results:
                        st.markdown(f"**工具**：{result.get('tool_name')}")
                        st.markdown(f"**输出**：{result.get('output')}")
                        st.markdown("")
            
            # 显示最终回答
            if "final_answer" in final_state:
                st.markdown("#### 3. 最终回答")
                answer = final_state["final_answer"].get("answer", "")
                if answer:
                    st.markdown(answer)
                sources = final_state["final_answer"].get("sources", [])
                if sources:
                    st.markdown(f"**来源**：{sources}")


def visualize_react():
    """可视化反应型智能体"""
    st.markdown("### 03 - 反应型智能体 (ReAct)")
    
    # 加载03_react模块
    spec = importlib.util.spec_from_file_location("react", "03_react.py")
    react = importlib.util.module_from_spec(spec)
    sys.modules["react"] = react
    spec.loader.exec_module(react)
    
    # 从模块中导入所需函数和类
    init_llm = react.init_llm
    build_app = react.build_app
    run_workflow = react.run_workflow
    
    # 用户输入区域
    default_request = "请计算表达式 12*(3+4)，并用一句话说明结果。"
    user_request = st.text_area("输入您的问题", value=default_request, height=100)
    
    # 执行按钮
    if st.button("开始执行ReAct工作流"):
        # 检查API密钥
        if not os.environ.get("MODELSCOPE_API_KEY"):
            st.error("请先设置API密钥")
        else:
            with st.spinner("正在初始化系统..."):
                # 初始化LLM
                llm = init_llm()
                
                # 构建工作流
                app = build_app(llm)
            
            st.success("系统初始化完成！")
            
            # 创建日志显示区域
            logs_container = st.empty()
            log_content = ""
            
            # 重定向控制台输出到日志区域
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # 执行工作流
                final_state = run_workflow(app, user_request)
            
            # 获取控制台输出
            log_content = f.getvalue()
            
            # 显示日志
            st.markdown("### 执行日志")
            st.text_area("", value=log_content, height=300, disabled=True)
            
            # 显示结果
            st.markdown("### 执行结果")
            
            # 显示步骤
            if "steps" in final_state:
                st.markdown("#### 1. ReAct 步骤")
                steps = final_state["steps"]
                for i, step in enumerate(steps):
                    st.markdown(f"**步骤 {i+1}**")
                    if step.get("thought"):
                        st.markdown(f"- 思考：{step.get('thought')}")
                    if step.get("action"):
                        st.markdown(f"- 行动：{step.get('action')}")
                        if step.get("action_input"):
                            st.markdown(f"- 输入：{step.get('action_input')}")
                    if step.get("observation"):
                        st.markdown(f"- 观察：{step.get('observation')}")
                    st.markdown("")
            
            # 显示最终回答
            if "final_answer" in final_state:
                st.markdown("#### 2. 最终回答")
                st.markdown(final_state["final_answer"])
            elif "steps" in final_state and steps and steps[-1].get("final_answer"):
                st.markdown("#### 2. 最终回答")
                st.markdown(steps[-1].get("final_answer"))


def visualize_planning():
    """可视化规划型智能体"""
    st.markdown("### 04 - 规划型智能体 (Planning)")
    
    # 加载04_planning.py模块
    spec = importlib.util.spec_from_file_location("planning", "04_planning.py")
    planning = importlib.util.module_from_spec(spec)
    sys.modules["planning"] = planning
    spec.loader.exec_module(planning)
    
    # 从模块中导入所需函数和类
    init_llm = planning.init_llm
    web_search = planning.web_search
    ModelScopeChatWithTools = planning.ModelScopeChatWithTools
    react_agent_app = planning.react_agent_app
    planning_agent_app = planning.planning_agent_app
    
    # 用户输入区域
    default_request = "查找北京、上海和广州的人口。然后计算它们的总人口。最后，将总人口与中国人口进行比较，并说明哪个更大。"
    user_request = st.text_area("输入您的请求", value=default_request, height=100)
    
    # 执行按钮
    if st.button("开始执行规划工作流"):
        # 检查API密钥
        if not os.environ.get("MODELSCOPE_API_KEY"):
            st.error("请先设置API密钥")
        else:
            # 创建日志显示区域
            logs_container = st.empty()
            log_content = ""
            
            # 重定向控制台输出到日志区域
            import io
            from contextlib import redirect_stdout
            from langchain_core.messages import HumanMessage
            
            f = io.StringIO()
            with redirect_stdout(f):
                # 执行工作流
                planning_result = planning_agent_app.invoke({
                    "messages": [
                        HumanMessage(content=user_request)
                    ]
                })
            
            # 获取控制台输出
            log_content = f.getvalue()
            
            # 显示日志
            st.markdown("### 执行日志")
            st.text_area("", value=log_content, height=300, disabled=True)
            
            # 显示结果
            st.markdown("### 执行结果")
            
            # 显示规划过程
            messages = planning_result["messages"]
            if messages:
                # 显示生成的计划
                for i, msg in enumerate(messages):
                    if i == 1 and hasattr(msg, 'content') and "1. " in msg.content and "2. " in msg.content:
                        st.markdown("#### 1. 生成的计划")
                        st.markdown(msg.content)
                        break
                
                # 显示执行结果和最终答案
                for i, msg in reversed(list(enumerate(messages))):
                    if hasattr(msg, 'content') and msg.content:
                        if "执行完成" in msg.content:
                            st.markdown("#### 2. 执行结果")
                            st.markdown(msg.content.replace("执行完成。结果：\n", ""))
                        elif (i == len(messages) - 1) or "最终答案" in msg.content:
                            st.markdown("#### 3. 最终答案")
                            st.markdown(msg.content)
                            break


def visualize_multi_agent():
    """可视化多智能体系统"""
    # 加载05_multi_agent模块
    spec = importlib.util.spec_from_file_location("multi_agent", "05_multi_agent.py")
    multi_agent = importlib.util.module_from_spec(spec)
    sys.modules["multi_agent"] = multi_agent
    spec.loader.exec_module(multi_agent)
    
    # 从动态加载的模块中导入所需函数和变量
    init_llm = multi_agent.init_llm
    web_search = multi_agent.web_search
    ModelScopeChatWithTools = multi_agent.ModelScopeChatWithTools
    build_monolithic_agent = multi_agent.build_monolithic_agent
    build_multi_agent_system = multi_agent.build_multi_agent_system
    GLOBAL_LOGS = multi_agent.GLOBAL_LOGS
    from langchain_core.messages import SystemMessage, HumanMessage
    
    st.markdown("### 05 - 多智能体系统 (Multi-Agent)")
    
    # 公司选择
    companies = [
        "NVIDIA (NVDA)",
        "阿里巴巴 (BABA)",
        "苹果 (AAPL)",
        "微软 (MSFT)",
        "特斯拉 (TSLA)",
        "亚马逊 (AMZN)"
    ]
    selected_company = st.selectbox("选择分析公司", companies)
    
    # 自定义公司选项
    custom_company = st.text_input("或输入自定义公司", "")
    if custom_company:
        selected_company = custom_company
    
    # 系统选择
    analysis_type = st.radio(
        "选择分析系统",
        ["多智能体系统", "单智能体系统", "对比分析"]
    )
    
    # 分析按钮
    if st.button("开始分析"):
        # 检查API密钥
        if not os.environ.get("MODELSCOPE_API_KEY"):
            st.error("请先设置API密钥")
        else:
            with st.spinner("正在初始化系统..."):
                # 初始化LLM和工具
                llm = init_llm()
                llm_with_tools = ModelScopeChatWithTools(llm, [web_search])
                
                # 构建单智能体系统
                monolithic_agent = build_monolithic_agent(llm_with_tools)
                
                # 构建多智能体系统
                multi_agent_system = build_multi_agent_system()
            
            st.success("系统初始化完成！")
            
            # 定义分析查询
            analysis_query = f"为{selected_company}创建一份简要但全面的市场分析报告。报告应包括三个部分：1. 最近新闻和市场情绪摘要。2. 股票价格趋势的基本技术分析。3. 公司最近财务表现的分析。"
            
            # 显示查询内容
            st.markdown(f"#### 分析任务：")
            st.info(analysis_query)
            
            # 执行分析
            if analysis_type in ["多智能体系统", "对比分析"]:
                st.markdown("## 🎯 多智能体系统分析结果")
                
                # 清空之前的日志
                GLOBAL_LOGS.clear()
                
                # 创建日志显示区域
                logs_container = st.empty()
                log_content = ""
                
                # 执行多智能体系统分析
                final_multi_output = multi_agent_system.invoke({
                    "messages": [
                        HumanMessage(content=analysis_query)
                    ]
                })
                
                # 显示所有日志
                for log in GLOBAL_LOGS:
                    log_content += f"{log}<br>"
                logs_container.markdown(f"### 执行日志<br><div class='log-section'>{log_content}</div>", unsafe_allow_html=True)
                
                # 展示多智能体分析结果
                messages = final_multi_output['messages']
                
                for msg in messages:
                    if hasattr(msg, 'content') and msg.content:
                        if "## 新闻与市场情绪分析" in msg.content:
                            st.markdown("### 📰 新闻与市场情绪分析")
                            st.markdown(msg.content.replace("## 新闻与市场情绪分析", ""))
                        elif "## 技术分析" in msg.content:
                            st.markdown("### 📈 技术分析")
                            st.markdown(msg.content.replace("## 技术分析", ""))
                        elif "## 财务分析" in msg.content:
                            st.markdown("### 💰 财务分析")
                            st.markdown(msg.content.replace("## 财务分析", ""))
                        else:
                            st.markdown("### 📋 最终综合报告")
                            st.markdown(msg.content)
                
                # 保存多智能体结果用于对比
                multi_agent_result = messages[-1].content if messages else ""
            
            if analysis_type in ["单智能体系统", "对比分析"]:
                # 添加分隔线
                if analysis_type == "对比分析":
                    st.markdown("---")
                
                st.markdown("## 🎯 单智能体系统分析结果")
                
                # 执行单智能体系统分析
                with st.spinner("单智能体系统正在分析..."):
                    final_mono_output = monolithic_agent.invoke({
                        "messages": [
                            SystemMessage(content="你是一位专业的金融分析师。你必须创建一份全面的报告，涵盖用户请求的所有方面。"),
                            HumanMessage(content=analysis_query)
                        ]
                    })
                
                # 展示单智能体分析结果
                mono_message = final_mono_output['messages'][-1].content
                st.markdown(mono_message)
                
                # 保存单智能体结果用于对比
                mono_agent_result = mono_message
            
            if analysis_type == "对比分析":
                st.markdown("---")
                st.markdown("## 📊 系统对比")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🤖 单智能体系统")
                    st.markdown("- **优点**：结构简单，单一入口")
                    st.markdown("- **缺点**：分析可能不够深入，各领域专业度有限")
                    st.markdown("- **适用场景**：简单任务，快速响应")
                
                with col2:
                    st.markdown("### 👥 多智能体系统")
                    st.markdown("- **优点**：各领域分析更深入，专业度更高，结果更全面")
                    st.markdown("- **缺点**：结构复杂，需要更多的协调和资源")
                    st.markdown("- **适用场景**：复杂任务，需要多领域专业知识")

def visualize_planner_executor_verifier():
    """可视化规划→执行→验证智能体"""
    st.markdown("### 06 - 规划→执行→验证智能体 (Planner→Executor→Verifier)")
    
    # 加载06_planner_executor_verifier模块
    spec = importlib.util.spec_from_file_location("planner_executor_verifier", "06_planner_executor_verifier.py")
    planner_executor_verifier = importlib.util.module_from_spec(spec)
    sys.modules["planner_executor_verifier"] = planner_executor_verifier
    spec.loader.exec_module(planner_executor_verifier)
    
    # 从模块中导入所需函数和类
    init_llm = planner_executor_verifier.init_llm
    build_app = planner_executor_verifier.build_app
    run_workflow = planner_executor_verifier.run_workflow
    print_execution_results = planner_executor_verifier.print_execution_results
    
    # 用户输入区域
    default_request = "查询苹果公司上一财年的研发支出和员工数量，计算人均研发支出"
    user_request = st.text_area("输入您的请求", value=default_request, height=100)
    
    # 执行按钮
    if st.button("开始执行规划→执行→验证工作流"):
        # 检查API密钥
        if not os.environ.get("MODELSCOPE_API_KEY"):
            st.error("请先设置API密钥")
        else:
            with st.spinner("正在初始化系统..."):
                # 初始化LLM
                llm = init_llm()
                
                # 构建工作流
                app = build_app(llm)
            
            st.success("系统初始化完成！")
            
            # 创建日志显示区域
            logs_container = st.empty()
            log_content = ""
            
            # 重定向控制台输出到日志区域
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # 执行工作流
                final_state = run_workflow(app, user_request)
                
            # 获取控制台输出
            log_content = f.getvalue()
            
            # 显示日志
            st.markdown("### 执行日志")
            st.text_area("", value=log_content, height=300, disabled=True)
            
            # 显示结果
            st.markdown("### 执行结果")
            
            # 显示执行过程
            if "intermediate_steps" in final_state:
                st.markdown("#### 1. 执行步骤")
                for i, step in enumerate(final_state["intermediate_steps"]):
                    st.markdown(f"**步骤 {i+1}**：{step}")
                    st.markdown("")
            
            # 显示最终答案
            if "final_answer" in final_state and final_state["final_answer"]:
                st.markdown("#### 2. 最终答案")
                st.markdown(final_state["final_answer"])


def visualize_blackboard_system():
    """可视化黑板系统"""
    st.markdown("### 07 - 黑板系统 (Blackboard System)")
    
    # 加载07_blackboard模块
    spec = importlib.util.spec_from_file_location("blackboard", "07_blackboard.py")
    blackboard = importlib.util.module_from_spec(spec)
    sys.modules["blackboard"] = blackboard
    spec.loader.exec_module(blackboard)
    
    # 从模块中导入所需函数和类
    init_llm = blackboard.init_llm
    build_blackboard_system = blackboard.build_blackboard_system
    run_blackboard_system = blackboard.run_blackboard_system
    search_tool = blackboard.search_tool
    
    # 用户输入区域
    default_request = "查找 NVIDIA 的最新重大新闻。根据该新闻的情绪，进行技术分析（如果新闻是中性或积极的）或财务分析（如果新闻是负面的）。"
    user_request = st.text_area("输入您的请求", value=default_request, height=100)
    
    # 执行按钮
    if st.button("开始执行黑板系统工作流"):
        # 检查API密钥
        if not os.environ.get("MODELSCOPE_API_KEY"):
            st.error("请先设置API密钥")
        else:
            with st.spinner("正在初始化系统..."):
                # 初始化LLM
                llm = init_llm()
                
                # 构建黑板系统
                blackboard_app = build_blackboard_system(llm, search_tool)
            
            st.success("系统初始化完成！")
            
            # 创建日志显示区域
            logs_container = st.empty()
            log_content = ""
            
            # 重定向控制台输出到日志区域
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # 执行工作流
                final_result = run_blackboard_system(blackboard_app, user_request)
                
            # 获取控制台输出
            log_content = f.getvalue()
            
            # 显示日志
            st.markdown("### 执行日志")
            st.text_area("", value=log_content, height=300, disabled=True)
            
            # 显示结果
            st.markdown("### 执行结果")
            
            # 显示最终报告
            for item in final_result["blackboard"]:
                if "[报告撰写者]" in item:
                    st.markdown("#### 1. 最终报告")
                    st.markdown(item.replace("[报告撰写者]", ""))
                    break
            
            # 显示信息板完整内容
            st.markdown("#### 2. 信息板完整内容")
            for i, item in enumerate(final_result["blackboard"]):
                st.markdown(f"**[{i+1}] {item.splitlines()[0]}**")
                content = "\n".join(item.splitlines()[1:])
                st.markdown(content)
                st.markdown("")

# 根据选择的架构显示不同的内容
if "01 - 反思型智能体" in selected_architecture:
    visualize_reflection()
elif "02 - 工具使用智能体" in selected_architecture:
    visualize_tool_use()
elif "03 - 反应型智能体" in selected_architecture:
    visualize_react()
elif "04 - 规划型智能体" in selected_architecture:
    visualize_planning()
elif "05 - 多智能体系统" in selected_architecture:
    visualize_multi_agent()
elif "06 - 规划→执行→验证智能体" in selected_architecture:
    visualize_planner_executor_verifier()
elif "07 - 黑板系统" in selected_architecture:
    visualize_blackboard_system()

# 页脚信息
st.markdown("---")
st.markdown("### 关于系统")
st.markdown("这是一个基于LangGraph构建的Agentic Architecture可视化系统，支持多种智能体架构的交互式分析。")
st.markdown("\n### 架构示例说明")
st.markdown("- **01 - 反思型智能体**：能够自我反思并改进输出的智能体")
st.markdown("- **02 - 工具使用智能体**：能够调用外部工具获取信息的智能体")
st.markdown("- **03 - 反应型智能体**：基于环境反馈做出反应的智能体")
st.markdown("- **04 - 规划型智能体**：能够制定和执行任务计划的智能体")
st.markdown("- **05 - 多智能体系统**：由多个专业智能体组成的协作系统")
st.markdown("- **06 - 规划→执行→验证智能体**：能够检测并纠正执行错误的智能体架构")
st.markdown("- **07 - 黑板系统**：多智能体协作的黑板系统，包含专家智能体和动态控制器")

st.markdown("\n### 技术栈")
st.markdown("- **LangGraph**：构建智能体工作流")
st.markdown("- **ModelScope**：提供语言模型支持")
st.markdown("- **Streamlit**：构建交互式界面")
st.markdown("- **Python**：主要开发语言")