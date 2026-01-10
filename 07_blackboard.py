# -*- coding: utf-8 -*-
"""
黑板系统（Blackboard System）的可运行示例

学习目标（看完你能做到什么）：
- 理解什么是黑板系统：一种高度灵活的多智能体协调模式，通过共享数据存储和动态控制器实现协作
- 掌握如何使用LangGraph构建黑板系统的工作流
- 学会设计动态控制器来决定下一个执行的智能体
- 对比线性多智能体系统和黑板系统的优缺点

核心概念速览：
- 共享内存（黑板）：中央数据存储，所有智能体可以读取和写入
- 专家智能体：具有特定专业知识的独立智能体
- 动态控制器：观察黑板状态并决定下一个执行的智能体
- 机会主义激活：智能体根据当前问题状态被动态激活

运行前准备：
- 项目根目录创建 `.env` 并配置：
  - `MODELSCOPE_API_KEY`（必需，ModelScope 推理令牌）
  - `MODELSCOPE_BASE_URL`（可选，默认：https://api-inference.modelscope.cn/v1）
  - `MODELSCOPE_MODEL_ID`（可选，默认：deepseek-ai/DeepSeek-V3.2）
  - `MODELSCOPE_MODEL_ID_R1`（备用模型，可选）
  - `LANGCHAIN_API_KEY`（可选，用于 LangSmith 追踪）
  - `TAVILY_API_KEY`（可选，用于真实网络搜索）

如何运行：
- 直接运行默认示例：`python 07_blackboard.py`
- 自定义查询：`python 07_blackboard.py --query "分析最近的阿里巴巴新闻并生成投资建议"`
- 跳过线性系统运行：`python 07_blackboard.py --no-sequential`
- 启用调试模式：`python 07_blackboard.py --debug`

阅读建议：
- 先从"数据结构与模型定义"开始理解黑板和线性系统的状态设计
- 然后看"LLM初始化"部分了解如何构建智能体使用的语言模型
- 接着学习"专家智能体"实现，理解每个角色的职责
- 最后看"工作流构建"了解如何将各个智能体组织起来协作
"""

import os
import re
import json
import argparse
from typing import List, Annotated, TypedDict, Optional

from dotenv import load_dotenv

from pydantic import BaseModel, Field

# LangChain components
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# LangGraph components
from langgraph.graph import StateGraph, END

# For pretty printing
from rich.console import Console
from rich.markdown import Markdown
from rich.logging import RichHandler

import logging
from openai import OpenAI, RateLimitError, APIError

# 全局控制台实例
console = Console()

# =========================
# 1) 数据结构与模型定义
# =========================

# 线性多智能体系统的状态
class SequentialState(TypedDict):
    """线性多智能体系统的共享状态"""
    user_request: str
    news_report: Optional[str]
    technical_report: Optional[str]
    financial_report: Optional[str]
    final_report: Optional[str]

# 黑板系统状态定义
class BlackboardState(TypedDict):
    """黑板系统的共享状态管理结构"""
    user_request: str
    blackboard: list[str]  # 共享信息板，所有智能体可读写
    available_agents: list[str]  # 可用的专家智能体列表，用于控制器选择下一个执行的智能体
    next_agent: Optional[str]  # 控制器决定的下一个智能体

# 控制器决策模型
class ControllerDecision(BaseModel):
    """控制器的决策结果"""
    next_agent: str = Field(description="下一个要调用的智能体名称或'FINISH'")
    reasoning: str = Field(description="做出此决策的原因")

# =========================
# 2) LLM 初始化
# =========================

class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口适配器:
    - 提供 invoke(prompt) 基本调用
    - 提供 with_structured_output(PydanticModel) 的结构化输出包装
    """
    def __init__(self, base_url: str, api_key: str, model: str, fallback_model: Optional[str] = None, temperature: float = 0.2, extra_body: Optional[dict] = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.fallback_model = fallback_model
        self.base_url = base_url
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.switched = False

    def invoke(self, prompt: str):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
                extra_body=self.extra_body,
            )
            return resp.choices[0].message.content
        except (RateLimitError, APIError) as e:
            if not self.switched and self.fallback_model:
                console.print(f"[bold yellow]⚠️ 主模型请求失败：{e}，尝试切换到备用模型[/bold yellow]")
                self.model = self.fallback_model
                self.switched = True
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=False,
                    extra_body=self.extra_body,
                )
                return resp.choices[0].message.content
            else:
                raise

    def with_structured_output(self, pyd_model: type[BaseModel]):
        class _StructuredWrapper:
            def __init__(self, outer: "ModelScopeChat"):
                self.outer = outer

            def invoke(self, prompt: str) -> BaseModel:
                # 通过系统提示约束仅输出 JSON
                schema = pyd_model.model_json_schema()
                props = schema.get("properties", {})
                required = schema.get("required", [])
                schema_text_lines = []
                for k, v in props.items():
                    t = v.get("type", "string")
                    schema_text_lines.append(f"- {k}: {t}")
                schema_text = "\n".join(schema_text_lines) or "- 请按模型定义生成字段"
                required_text = ", ".join(required) if required else "所有字段"
                system_msg = (
                    "你是一个结构化输出生成器。只输出一个 JSON 对象，严格匹配以下字段与类型：\n"
                    f"{schema_text}\n"
                    f"必须包含字段：{required_text}\n"
                    "不要输出任何解释或多余文本（例如代码块标记、前后缀）。"
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                try:
                    resp = self.outer.client.chat.completions.create(
                        model=self.outer.model,
                        messages=messages,
                        temperature=self.outer.temperature,
                        stream=False,
                        extra_body=self.outer.extra_body,
                    )
                    content = resp.choices[0].message.content or ""
                except (RateLimitError, APIError) as e:
                    if not self.outer.switched and self.outer.fallback_model:
                        console.print(f"[bold yellow]⚠️ 主模型请求失败：{e}，尝试切换到备用模型[/bold yellow]")
                        self.outer.model = self.outer.fallback_model
                        self.outer.switched = True
                        resp = self.outer.client.chat.completions.create(
                            model=self.outer.model,
                            messages=messages,
                            temperature=self.outer.temperature,
                            stream=False,
                            extra_body=self.outer.extra_body,
                        )
                        content = resp.choices[0].message.content or ""
                    else:
                        raise
                
                # 提取并解析 JSON
                def _extract_json(s: str) -> str:
                    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', s)
                    return m.group(1) if m else "{}"
                raw = content.strip()
                try:
                    data = json.loads(raw)
                except Exception:
                    data = json.loads(_extract_json(raw))
                
                # 解析为 Pydantic 模型
                try:
                    parsed = pyd_model.model_validate(data)
                    return parsed
                except Exception:
                    # 尝试字段映射
                    if "code" not in data and "function" in data:
                        data["code"] = data.pop("function")
                    if "explanation" not in data and "desc" in data:
                        data["explanation"] = data.pop("desc")
                    parsed = pyd_model.model_validate(data)
                    return parsed

        return _StructuredWrapper(self)

class ModelScopeChatWithTools:
    """
    ModelScopeChat 的工具调用包装器：
    - 支持将工具绑定到 LLM
    - 处理工具调用请求和响应
    """
    def __init__(self, llm_instance: ModelScopeChat, tools: list):
        self.llm = llm_instance
        self.tools = tools
    
    def invoke(self, messages: list):
        # 将消息转换为提示字符串
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"[系统] {msg.content}\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"[用户] {msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt += f"[助手] {msg.content}\n"
            else:
                try:
                    prompt += f"[消息] {msg.content}\n"
                except:
                    continue
        
        # 生成响应
        response = self.llm.invoke(prompt)
        
        # 解析工具调用
        tool_calls = []
        if "web_search" in response and "query" in response:
            try:
                tool_data = json.loads(response)
                if "tool_call" in tool_data:
                    tool_calls = [{
                        "name": tool_data["tool_call"]["name"],
                        "args": tool_data["tool_call"]["args"]
                    }]
            except:
                # 检查简单模式
                match = re.search(r'web_search\(query=[\'"]([^\'"]+)[\'"]\)', response)
                if match:
                    tool_calls = [{
                        "name": "web_search",
                        "args": {"query": match.group(1)}
                    }]
        
        # 执行工具调用
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # 查找并执行工具
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_result = tool.invoke(tool_args)
                        
                        # 将工具结果添加到提示
                        prompt += f"[工具] {tool_name}\n{tool_result}\n"
                        
                        # 重新生成响应
                        response = self.llm.invoke(prompt)
                        break
        
        return AIMessage(content=response, tool_calls=tool_calls)

def init_llm() -> ModelScopeChat:
    """
    初始化 ModelScope LLM（OpenAI 兼容接口）。
    """
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    fallback_model_id = os.environ.get("MODELSCOPE_MODEL_ID_R1")
    
    extra = {
        "enable_thinking": True,
        "trust_request_chat_template": True,
        "response_format": {"type": "json_object"}
    }
    
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, fallback_model=fallback_model_id, temperature=0.2, extra_body=extra)

# =========================
# 3) 工具定义
# =========================

try:
    # 尝试初始化真实的 Tavily 搜索工具
    if os.environ.get("TAVILY_API_KEY"):
        search_tool = TavilySearch(max_results=2)
    else:
        # 如果没有 API 密钥，使用模拟搜索工具
        @tool
        def web_search(query: str) -> str:
            """模拟网络搜索工具"""
            console = Console()
            console.print(f"--- 工具：搜索 '{query}'...")
            
            # 预定义的模拟数据
            mock_data = {
                "NVIDIA 最近新闻": "NVIDIA 发布了新一代 AI 芯片 H200，性能比上一代提升 2.5 倍。同时，NVIDIA 与多家云服务提供商达成合作，扩大其 AI 基础设施业务。",
                "NVIDIA 股价趋势": "NVIDIA 股价在过去 6 个月上涨了 85%，主要受 AI 芯片需求激增和良好的财报表现推动。技术指标显示，该股目前处于强势上涨趋势中。",
                "NVIDIA 财务表现": "NVIDIA 2023 财年第四季度营收达到 221 亿美元，同比增长 265%；净利润为 123 亿美元，同比增长 769%。数据中心业务收入占总营收的 70% 以上。",
                "阿里巴巴 最近新闻": "阿里巴巴宣布进行组织架构调整，成立六大业务集团，各集团将独立融资和上市。同时，阿里巴巴加大对AI技术的投入，推出通义千问大模型。",
                "阿里巴巴 股价趋势": "阿里巴巴股价在过去 6 个月下跌了 15%，主要受宏观经济环境和监管政策影响。技术指标显示，该股目前处于盘整阶段。",
                "阿里巴巴 财务表现": "阿里巴巴2024财年第二季度营收达2247亿元人民币，同比增长9%；净利润达488亿元人民币，同比增长19%。云计算业务收入达276亿元人民币，同比增长2%。",
            }
            
            for key, value in mock_data.items():
                if key in query:
                    return value
            
            return f"模拟搜索结果：{query}"
        
        search_tool = web_search
except Exception as e:
    # 如果初始化失败，使用模拟工具
    console.print(f"[bold yellow]⚠️ 初始化Tavily搜索工具失败，使用模拟工具：{e}[/bold yellow]")
    
    @tool
    def web_search(query: str) -> str:
        """模拟网络搜索工具"""
        console = Console()
        console.print(f"--- 工具：搜索 '{query}'...")
        return f"模拟搜索结果：{query}"
    
    search_tool = web_search

# =========================
# 4) 线性多智能体系统
# =========================

def build_sequential_system(llm: ModelScopeChat, search_tool):
    """构建线性多智能体系统（作为基线对比）"""
    
    def news_analyst_node_seq(state: SequentialState):
        """新闻分析师智能体：分析最新新闻"""
        console.print("--- (线性系统) 调用新闻分析师 ---")
        prompt = f"你的任务是作为专业的新闻分析师。查找用户请求中主题的最新重大新闻并提供简洁摘要。\n\n用户请求：{state['user_request']}"
        agent = ModelScopeChatWithTools(llm, [search_tool])
        result = agent.invoke([HumanMessage(content=prompt)])
        return {"news_report": result.content}
    
    def technical_analyst_node_seq(state: SequentialState):
        """技术分析师智能体：基于新闻进行技术分析"""
        console.print("--- (线性系统) 调用技术分析师 ---")
        prompt = f"你的任务是作为专业的技术分析师。基于以下新闻报道，对公司股票进行技术分析。\n\n新闻报道：\n{state['news_report']}"
        agent = ModelScopeChatWithTools(llm, [search_tool])
        result = agent.invoke([HumanMessage(content=prompt)])
        return {"technical_report": result.content}
    
    def financial_analyst_node_seq(state: SequentialState):
        """财务分析师智能体：基于新闻分析财务表现"""
        console.print("--- (线性系统) 调用财务分析师 ---")
        prompt = f"你的任务是作为专业的财务分析师。基于以下新闻报道，分析公司最近的财务表现。\n\n新闻报道：\n{state['news_report']}"
        agent = ModelScopeChatWithTools(llm, [search_tool])
        result = agent.invoke([HumanMessage(content=prompt)])
        return {"financial_report": result.content}
    
    def report_writer_node_seq(state: SequentialState):
        """报告撰写者智能体：综合所有分析生成最终报告"""
        console.print("--- (线性系统) 调用报告撰写者 ---")
        prompt = f"""你是一名专业的报告撰写者。你的任务是将新闻、技术和财务分析师的信息综合成一份单一、连贯的报告，直接回答用户的原始请求。

用户请求：{state['user_request']}

以下是要合并的报告：
---
新闻报道：{state['news_report']}
---
技术分析：{state['technical_report']}
---
财务分析：{state['financial_report']}
"""
        report = llm.invoke(prompt)
        return {"final_report": report}
    
    # 构建线性图
    seq_graph_builder = StateGraph(SequentialState)
    seq_graph_builder.add_node("news", news_analyst_node_seq)
    seq_graph_builder.add_node("tech", technical_analyst_node_seq)
    seq_graph_builder.add_node("finance", financial_analyst_node_seq)
    seq_graph_builder.add_node("writer", report_writer_node_seq)
    
    # 固定的、硬编码的序列
    seq_graph_builder.set_entry_point("news")
    seq_graph_builder.add_edge("news", "tech")
    seq_graph_builder.add_edge("tech", "finance")
    seq_graph_builder.add_edge("finance", "writer")
    seq_graph_builder.add_edge("writer", END)
    
    return seq_graph_builder.compile()

def run_sequential_system(app, query):
    """运行线性多智能体系统"""
    console.print("\n" + "="*60)
    console.print("[bold cyan]线性多智能体系统运行开始[/bold cyan]")
    console.print("="*60)
    
    result = app.invoke({"user_request": query})
    
    console.print("\n--- 线性系统最终报告 ---")
    console.print(Markdown(result["final_report"]))
    
    return result

# =========================
# 5) 黑板系统
# =========================

def create_blackboard_specialist(llm, search_tool, agent_name, description):
    """创建黑板系统的专家智能体工厂函数"""
    def specialist_node(state: BlackboardState):
        console.print(f"--- (黑板系统) 调用 {agent_name} ---")
        
        # 构建信息板内容
        blackboard_content = "\n\n".join(state["blackboard"]) if state["blackboard"] else "信息板目前为空。"
        
        # 构建提示
        prompt = f"""你是专业的{description}。你的任务是：
1. 阅读用户原始请求
2. 分析当前信息板上的已有信息
3. 执行你的专业任务
4. 将分析结果添加到信息板

用户原始请求：{state['user_request']}

当前信息板内容：
---
{blackboard_content}
---

请提供专业分析结果，确保内容简洁明了，直接添加到信息板。
"""
        
        # 调用智能体
        agent = ModelScopeChatWithTools(llm, [search_tool])
        result = agent.invoke([HumanMessage(content=prompt)])
        
        # 将分析结果添加到信息板
        new_blackboard = state["blackboard"].copy()
        new_blackboard.append(f"[{agent_name}]\n{result.content}")
        
        return {"blackboard": new_blackboard}
    
    return specialist_node

def controller_node(state: BlackboardState, llm: ModelScopeChat):
    """黑板系统的中央控制器：决定下一个执行的智能体
    
    Args:
        state: 黑板系统的当前状态
        llm: 语言模型实例，用于生成决策
        
    Returns:
        更新后的状态，包含下一个要执行的智能体名称
    """
    console.print("--- 控制器：分析黑板... ---")
    
    # 使用结构化输出的语言模型，确保决策结果符合ControllerDecision格式
    controller_llm = llm.with_structured_output(ControllerDecision)
    
    # 准备黑板内容供控制器分析
    blackboard_content = "\n\n".join(state["blackboard"]) if state["blackboard"] else "信息板目前为空。"
    # 获取可用智能体列表：这个列表在系统初始化时设定（见run_blackboard_system函数）
    agent_list = state["available_agents"]
    
    # 构建提示词，指导控制器做出决策
    prompt = f"""你是多智能体系统的中央控制器。你的工作是分析共享黑板和原始用户请求，决定下一步应该运行哪个专家智能体。

**原始用户请求：**
{state['user_request']}

**当前信息板内容：**
---
{blackboard_content}
---

**可用的专家智能体：**
{', '.join(agent_list)}

**你的任务：**
1. 仔细阅读用户请求和当前黑板内容
2. 确定下一步逻辑步骤，以更接近完整答案
3. 从可用智能体列表中选择一个最适合执行该步骤的智能体
4. 如果用户的请求已完全解决并且已经编写了最终报告，请选择'FINISH'

请返回你的决策和理由。
"""
    
    # 调用语言模型生成决策
    decision_result = controller_llm.invoke(prompt)
    console.print(f"--- 控制器：决定调用 '{decision_result.next_agent}'。原因：{decision_result.reasoning} ---")
    
    # 返回只包含next_agent的更新状态
    # 这个值会被router函数用于决定下一个执行的节点
    return {"next_agent": decision_result.next_agent}

def build_blackboard_system(llm: ModelScopeChat, search_tool):
    """构建黑板系统
    
    Args:
        llm: 语言模型实例
        search_tool: 搜索工具实例
        
    Returns:
        编译后的黑板系统工作流图
    """
    # 创建专家智能体
    news_analyst = create_blackboard_specialist(llm, search_tool, "新闻分析师", "新闻分析师，负责查找和分析最新新闻")
    technical_analyst = create_blackboard_specialist(llm, search_tool, "技术分析师", "技术分析师，负责股票技术分析")
    financial_analyst = create_blackboard_specialist(llm, search_tool, "财务分析师", "财务分析师，负责公司财务表现分析")
    report_writer = create_blackboard_specialist(llm, search_tool, "报告撰写者", "报告撰写者，负责综合所有信息生成最终报告")
    
    # 构建黑板图：使用StateGraph创建有状态工作流
    # BlackboardState是工作流的共享状态类型
    blackboard_graph_builder = StateGraph(BlackboardState)
    
    # 添加节点：每个节点代表一个智能体或控制器
    # 节点函数会接收当前状态并返回更新后的状态
    blackboard_graph_builder.add_node("news_analyst", news_analyst)          # 新闻分析师节点
    blackboard_graph_builder.add_node("technical_analyst", technical_analyst)  # 技术分析师节点
    blackboard_graph_builder.add_node("financial_analyst", financial_analyst)  # 财务分析师节点
    blackboard_graph_builder.add_node("report_writer", report_writer)        # 报告撰写者节点
    
    # 控制器节点：使用lambda函数封装controller_node
    # StateGraph要求节点函数只接受state一个参数
    # 但controller_node需要state和llm两个参数
    # 所以使用lambda创建闭包，将llm作为固定参数传递
    blackboard_graph_builder.add_node("controller", lambda state: controller_node(state, llm))
    
    # 设置入口点
    blackboard_graph_builder.set_entry_point("controller")
    
    # 定义条件边：根据控制器的决策路由
    def router(state: BlackboardState):
        next_agent = state["next_agent"]
        if next_agent == "FINISH":
            return END
        return next_agent
    
    # 添加边
    blackboard_graph_builder.add_conditional_edges("controller", router)
    blackboard_graph_builder.add_edge("news_analyst", "controller")
    blackboard_graph_builder.add_edge("technical_analyst", "controller")
    blackboard_graph_builder.add_edge("financial_analyst", "controller")
    blackboard_graph_builder.add_edge("report_writer", "controller")
    
    return blackboard_graph_builder.compile()

def run_blackboard_system(app, query):
    """运行黑板系统"""
    console.print("\n" + "="*60)
    console.print("[bold cyan]黑板系统运行启动[/bold cyan]")
    console.print("="*60)
    
    # 初始状态：为黑板系统设置初始参数
    initial_state = {
        "user_request": query,                 # 用户的原始查询
        "blackboard": [],                       # 初始黑板为空列表
        "available_agents": ["news_analyst", "technical_analyst", "financial_analyst", "report_writer"],  # 初始化所有可用的专家智能体
        "next_agent": None                      # 初始时没有下一个智能体
    }
    
    result = app.invoke(initial_state)
    
    console.print("\n--- 黑板系统最终结果 ---")
    for item in result["blackboard"]:
        if "[报告撰写者]" in item:
            console.print(Markdown(item.replace("[报告撰写者]", "")))
            break
    
    console.print("\n--- 信息板完整内容 ---")
    for i, item in enumerate(result["blackboard"]):
        console.print(f"\n[{i+1}] {item.splitlines()[0]}")
    
    return result

# =========================
# 6) 工具包装器和辅助函数
# =========================

def extract_company_name_from_query(query: str) -> str:
    """
    从查询字符串中提取公司名称
    
    参数：
        query: 用户查询字符串
    
    返回：
        提取的公司名称
    """
    import re
    
    # 查找常见的公司名称模式
    patterns = [
        r'(NVIDIA|英伟达|Alibaba|阿里巴巴|Tencent|腾讯|Baidu|百度|Microsoft|微软|Apple|苹果)',
        r'关于(.*?)的',
        r'分析(.*?)的',
        r'(.*?)最近'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # 如果是第一个模式，返回匹配的公司名称
            if '|' in pattern:
                return match.group(1)
            # 否则返回捕获组的内容
            return match.group(1).strip()
    
    # 默认返回 NVIDIA
    return "NVIDIA"


def format_blackboard_content(blackboard: list[str]) -> str:
    """
    格式化信息板内容以便展示
    
    参数：
        blackboard: 信息板内容列表
    
    返回：
        格式化后的字符串
    """
    if not blackboard:
        return "信息板目前为空。"
    
    formatted = "\n".join([f"[{i+1}] {item.splitlines()[0]}" for i, item in enumerate(blackboard)])
    return formatted


# =========================
# 7) 命令行接口和主函数
# =========================

def parse_args():
    """解析命令行参数
    
    命令行参数说明：
    - --query: 自定义查询内容，默认值为NVIDIA相关新闻分析请求
    - --no-sequential: 布尔选项，用于跳过线性多智能体系统的运行
    - --debug: 布尔选项，用于启用调试模式，显示更详细的运行信息
    
    action参数说明：
    - action="store_true": 用于定义布尔类型的命令行选项
    - 当用户在命令行中指定该选项（如--no-sequential）时，对应的变量会被设置为True
    - 不指定时，默认值为False
    
    使用示例：
    python 07_blackboard.py                  # 使用默认查询运行完整系统
    python 07_blackboard.py --query "分析阿里巴巴的最新动态"  # 使用自定义查询
    python 07_blackboard.py --no-sequential   # 只运行黑板系统，跳过线性系统
    python 07_blackboard.py --debug           # 启用调试模式运行
    python 07_blackboard.py --query "分析腾讯" --no-sequential --debug  # 组合使用多个参数
    """
    parser = argparse.ArgumentParser(description="黑板系统多智能体演示")
    
    # 定义字符串类型参数，用于接收用户的自定义查询
    parser.add_argument("--query", 
                        type=str, 
                        default="查找 NVIDIA 的最新重大新闻。根据该新闻的情绪，进行技术分析（如果新闻是中性或积极的）或财务分析（如果新闻是负面的）。",
                        help="自定义查询内容")
    
    # 定义布尔类型参数，使用action="store_true"，不接收额外值
    parser.add_argument("--no-sequential", 
                        action="store_true", 
                        help="跳过线性系统运行")
    
    # 定义布尔类型参数，用于启用调试模式
    parser.add_argument("--debug", 
                        action="store_true", 
                        help="启用调试模式")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()
    
    # 解析命令行参数
    args = parse_args()
    
    # 初始化控制台和日志
    global console
    console = Console()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler(console=console)])
    
    # 初始化 LLM
    console.print("--- 初始化语言模型... ---\n")
    llm = init_llm()
    
    # 构建系统
    console.print("--- 构建多智能体系统... ---\n")
    sequential_app = build_sequential_system(llm, search_tool)
    blackboard_app = build_blackboard_system(llm, search_tool)
    
    # 运行系统
    if not args.no_sequential:
        run_sequential_system(sequential_app, args.query)
    
    run_blackboard_system(blackboard_app, args.query)
    
    console.print("\n" + "="*60)
    console.print("[bold green]所有系统运行完成[/bold green]")
    console.print("="*60)
    console.print("\n[bold yellow]学习提示：[/bold yellow]")
    console.print("1. 对比线性系统和黑板系统的运行过程，理解它们的区别")
    console.print("2. 查看控制器的决策逻辑，思考如何优化它")
    console.print("3. 尝试修改查询内容，观察黑板系统的动态调整能力")

if __name__ == "__main__":
    main()
                