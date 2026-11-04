import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM

load_dotenv()

# ==================== 1. 初始化大模型 ====================
# 让 CrewAI 使用通义千问
llm = LLM(
    model="openai/qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.2
)


# ==================== 2. 定义 AI 团队角色 (Agents) ====================

# 角色一：产品经理
pm_agent = Agent(
    role='资深产品经理',
    goal='根据用户需求，设计出清晰、逻辑严密的软件功能需求文档。',
    backstory='你在大厂做了10年产品经理，非常擅长把模糊的需求转化为具体的开发任务。',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 角色二：高级开发工程师
dev_agent = Agent(
    role='Python高级开发工程师',
    goal='根据产品经理的需求文档，编写高质量、无Bug的 Python 代码。',
    backstory='你精通 Python 和各类算法，写出的代码极其简洁高效。',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# 角色三：测试工程师
qa_agent = Agent(
    role='测试工程师',
    goal='审查开发写的代码，找出潜在的 Bug，并给出修改建议。',
    backstory='你非常挑剔，绝对不会放过任何一个边界情况（Edge Case）。',
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# ==================== 3. 定义任务 (Tasks) ====================

# 用户需求
user_requirement = "开发一个 Python 函数，用于计算一个数字列表中的中位数。"

# 任务一：产品经理写需求
task1 = Task(
    description=f"根据以下用户需求，编写一份简短的需求分析文档：{user_requirement}",
    expected_output="包含功能描述、输入输出格式的需求文档。",
    agent=pm_agent
)

# 任务二：开发工程师写代码
task2 = Task(
    description="根据产品经理的需求文档，编写 Python 代码实现该功能。请加上必要的注释。",
    expected_output="一段完整的、可运行的 Python 代码。",
    agent=dev_agent,
    context=[task1] # 依赖 task1 的产出
)

# 任务三：测试工程师写测试用例
task3 = Task(
    description="审查上述 Python 代码，编写一段测试用例（使用 assert），确保代码逻辑在各种边界条件下（如空列表、偶数个元素）都没问题。",
    expected_output="包含测试用例的 Python 代码，并附带测试结果评估。",
    agent=qa_agent,
    context=[task2] # 依赖 task2 的产出
)

# ==================== 4. 组建团队并运行 ====================

dev_team = Crew(
    agents=[pm_agent, dev_agent, qa_agent],
    tasks=[task1, task2, task3],
    process=Process.sequential # 顺序执行：产品->开发->测试
)

if __name__ == "__main__":
    print("🚀 AI 开发团队已就位，开始工作...\n")
    result = dev_team.kickoff()
    print("\n" + "="*50)
    print("✅ 最终交付结果：\n")
    print(result)