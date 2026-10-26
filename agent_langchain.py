import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import requests
from datetime import datetime

load_dotenv()

# ==================== 1. 初始化大模型 ====================
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0
)


# ==================== 2. 定义工具 ====================

@tool
def add(a: float, b: float) -> float:
    """计算两个数字的和。当用户需要做加法时使用此工具。"""
    return a + b


@tool
def get_time() -> str:
    """获取当前系统的真实日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """查询指定城市的真实天气。输入必须是中文城市名，如：南宁。"""
    CITY_ID = {
        "北京": "101010100", "上海": "101020100", "广州": "101280101",
        "深圳": "101280601", "天津": "101030100", "南宁": "101300101"
    }
    city_id = CITY_ID.get(city)
    if not city_id:
        return f"抱歉，暂不支持查询{city}的天气。"

    url = "https://p84wcwqxc9.re.qweatherapi.com/v7/weather/now"
    params = {"location": city_id, "key": os.getenv("QWEATHER_API_KEY")}

    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("code") == "200":
            now = data["now"]
            return f"{city}当前天气：{now['text']}，气温 {now['temp']}℃。"
        return "天气查询失败。"
    except Exception as e:
        return f"天气查询出错：{e}"


# 把工具打包成列表
tools = [add, get_time, get_weather]

# ==================== 3. 构建 Agent 的提示词 ====================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个得力的 AI 助手，可以调用工具帮用户解决问题。回答要简洁。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ==================== 4. 构建 Agent ====================
# create_openai_tools_agent 是 LangChain 封装好的，直接帮你把 tools 和 prompt 结合
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ==================== 5. 运行测试 ====================
if __name__ == "__main__":
    print("LangChain Agent 已启动！输入 exit 退出。")
    while True:
        user_input = input("\n你：")
        if user_input.lower() in ["exit", "退出"]:
            break

        # 调用 Agent，LangChain 会自动处理：模型判断 -> 调工具 -> 拿结果 -> 再回答
        response = agent_executor.invoke({"input": user_input})
        print(f"\nAI：{response['output']}")