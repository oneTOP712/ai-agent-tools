import os
import json
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
#阿里云
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
#和风天气
qweather_key = os.getenv("QWEATHER_API_KEY")
# ==================== 工具定义 ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个数字的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数字"},
                    "b": {"type": "number", "description": "第二个数字"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，比如北京"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ==================== 工具函数 ====================
def add(a, b):
    return a + b


# 城市名 → 和风天气城市ID
CITY_ID = {
    "北京": "101010100",
    "上海": "101020100",
    "广州": "101280101",
    "深圳": "101280601",
    "天津": "101030100",
    "南宁": "101300101",
    "杭州": "101210101",
    "成都": "101270101",
}

def get_weather(city):
    city_id = CITY_ID.get(city)
    if not city_id:
        return f"暂时没有{city}的天气数据"

    url = "https://p84wcwqxc9.re.qweatherapi.com/v7/weather/now"
    params = {
        "location": city_id,
        "key": qweather_key
    }

    response = requests.get(url, params=params)
    data = response.json()


    if data.get("code") != "200":
        return f"查询失败，错误码：{data['code']}"

    now = data["now"]
    return f"{city}：{now['text']}，{now['temp']}℃"

def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 工具名 → 函数 的映射
func_map = {
    "add": add,
    "get_weather": get_weather,
    "get_time": get_time
}

# ==================== 对话主流程 ====================
def chat(user_input):
    messages = [{"role": "user", "content": user_input}]

    # 第一次调用：模型判断是否用工具
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
    )

    tool_call = response.choices[0].message.tool_calls

    if tool_call:
        # 记录模型的工具请求
        messages.append(response.choices[0].message)

        # 执行工具
        func_name = tool_call[0].function.name
        args = json.loads(tool_call[0].function.arguments)
        result = func_map[func_name](**args)

        # 记录工具结果
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call[0].id,
            "content": str(result)
        })

        # 第二次调用：模型回答
        final_response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        print(f"AI：{final_response.choices[0].message.content}")
    else:
        # 不需要工具，直接回答
        print(f"AI：{response.choices[0].message.content}")

# ==================== 主循环 ====================
print("输入 exit 退出")
while True:
    user_input = input("\n你：")
    if user_input.lower() in ["exit", "退出", "quit"]:
        print("再见！")
        break
    chat(user_input)