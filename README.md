# AI Agent & LLM Applications

本项目是基于 Python 与大语言模型（LLM）的 AI 应用开发实践集合。涵盖了从底层原生 SDK 到主流框架（LangChain, CrewAI）的落地实践，包含 AI Agent 工具调用、企业级 RAG 知识库、多智能体协作等核心场景。

## 🚀 核心项目模块

### 1. 企业级 RAG 私有知识库问答系统 (`rag_streamlit.py`)
*   **技术栈**：Streamlit, FastAPI, LangChain, FAISS, BM25, 通义千问 API
*   **功能亮点**：
    *   **前后端分离与全栈部署**：前端基于 Streamlit 搭建 Web 界面，已成功部署至云端；后端使用 FastAPI 提供流式服务。
    *   **混合检索（Hybrid Search）**：引入 `BM25Retriever` 与 `FAISS` 向量检索，通过 `EnsembleRetriever` 进行融合，解决纯向量检索对专有名词和精准数字不敏感的痛点。
    *   **多轮记忆与防幻觉**：使用 `History-Aware Retriever` 重写用户追问，结合 Prompt 约束，有效缓解大模型幻觉。
*   **在线体验**：[点击访问公网 Demo](https://ai-chenpingan-agent.streamlit.app)

### 2. 基于 CrewAI 的多智能体协作系统 (`multi_agent_dev.py`)
*   **技术栈**：Python, CrewAI, 通义千问 API
*   **功能亮点**：
    *   模拟真实软件工程团队，设计了产品经理、开发工程师、测试工程师三个独立角色 Agent。
    *   通过 `Process.sequential` 机制编排流水线，利用 `context` 依赖实现上下文传递，完成从“需求分析 -> 代码编写 -> 边界测试”的全自动闭环。

### 3. AI Agent 工具调用系统（双版本）
*   **底层原生版 (`tool.py`)**：通过 OpenAI 原生 SDK 手写 `tool_calls` 解析与回传，深刻理解大模型工具调用的底层流转机制。
*   **LangChain 框架版 (`agent_langchain.py`)**：使用 `@tool` 装饰器定义工具，通过 `AgentExecutor` 实现自动化决策与多轮循环调度。
*   **API 服务化 (`server.py`)**：基于 FastAPI 构建流式接口（`StreamingResponse`），实现打字机级别的对话体验。

## 🛠️ 技术栈

*   **语言**：Python
*   **大模型与框架**：通义千问 (Qwen), OpenAI SDK, LangChain, CrewAI
*   **检索与存储**：FAISS (向量库), BM25 (关键词检索)
*   **后端与部署**：FastAPI, Streamlit, Uvicorn

## ⚙️ 环境安装与运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
