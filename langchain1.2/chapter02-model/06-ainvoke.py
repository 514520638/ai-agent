# 异步调用（ainvoke）
# 说明：MiniMax（海螺 AI）兼容 OpenAI 协议，所以用 ChatOpenAI 作为客户端
#       不要用 init_chat_model —— 它不认识 MiniMax 供应商
import os
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="MiniMax-M3",
    api_key=os.getenv("MINIMAX_API_KEY"),  # 从环境变量读取，不要硬编码
    base_url="https://api.minimaxi.com/v1",
)

async def invoke_async():
    print("开始异步调用...")
    async_task = asyncio.create_task(llm.ainvoke("你叫什么名字？"))
    for i in range(3):
        await asyncio.sleep(1)
        print(f"异步调用进度：{i+1}/3")
 
    response = await async_task
    print(f"异步调用结果：{response.content}")


if __name__ == "__main__":
    asyncio.run(invoke_async())
