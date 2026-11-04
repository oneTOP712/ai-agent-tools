# AI Agent & LLM Applications

基于 Python 和大语言模型（LLM）的 AI 应用开发实践集合。涵盖了从底层原生 SDK 调用到主流框架（LangChain, CrewAI），以及 RAG 检索增强生成的完整落地项目。

## 🚀 核心项目

1. **原生 & LangChain Agent 工具调用** (`tool.py` / `agent_langchain.py`)
   - 分别使用 OpenAI 原生 SDK 和 LangChain 框架实现 Function Calling。
   - 支持算术计算、真实天气查询、时间获取等工具调度。

2. **Streamlit RAG 私有知识库问答系统** (`rag_streamlit.py`)
   - 带 Web 交互界面的 RAG 应用。
   - 使用 FAISS 构建本地向量库，支持 PDF/TXT 文档上传、文本切片、向量化检索及多轮记忆问答。

3. **CrewAI 多智能体协作系统** (`multi_agent_dev.py`)
   - 模拟软件开发团队流水线。
   - 编排产品经理、开发工程师、测试工程师三个独立 Agent，通过 `Process.sequential` 自动完成从需求分析到代码编写、边界测试的闭环。

## 🛠️ 技术栈

- **语言**：Python
- **大模型**：通义千问 (Qwen)
- **框架**：LangChain, CrewAI
- **向量库**：FAISS
- **前端与部署**：Streamlit
- **其他工具**：OpenAI SDK, 和风天气 API

## ⚙️ 环境安装与运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
