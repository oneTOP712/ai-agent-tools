import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

messages = [
    {"role": "system", "content": "你是一个打三角洲行动这个游戏非常厉害的顶级高手，起装备改枪，道具思路都很全面的高手。"}
]

print("输入 exit 退出")

while True:
    user_input = input("\n你：")
    if user_input.lower() in ["exit","退出","quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages
    )

    reply = response.choices[0].message.content
    print(f"\nAI：{reply}")

    messages.append({"role": "assistant", "content": reply})