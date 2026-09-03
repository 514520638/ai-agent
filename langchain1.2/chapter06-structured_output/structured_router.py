"""
structured_router.py
====================
智能路由：自动判断模型是否支持原生 tool calling，
        选择 with_structured_output 或 PydanticOutputParser。

使用：
    from structured_router import StructuredRouter

    router = StructuredRouter(chat_model=model)
    chain  = router.build(Person)
    person = chain.invoke("张三是一名30岁的软件工程师")
"""

from __future__ import annotations

import re
from typing import Any, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 1. 探测模型是否"真的"支持 tool calling
# ---------------------------------------------------------------------------
def detect_tool_calling_support(model: BaseChatModel, verbose: bool = True) -> bool:
    """
    用一次极轻量的探测调用判断模型是否真的支持 OpenAI 风格的 tool calling。

    判断逻辑：
        1) 检查 model.bind_tools 是否被重写（基类默认是占位）
        2) 绑一个最小 tool，要求 tool_choice="any"
        3) 看返回的 AIMessage.additional_kwargs 里有没有带 arguments 的 tool_calls
    """
    if type(model).bind_tools is BaseChatModel.bind_tools:
        if verbose:
            print("[router] 探针：模型类未重写 bind_tools，直接判定不支持")
        return False

    probe_tool = {
        "type": "function",
        "function": {
            "name": "_probe",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    try:
        bound = model.bind_tools([probe_tool], tool_choice="any")
        resp = bound.invoke("ping")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[router] 探针：bind_tools/invoke 抛异常 → 不支持。err={e!r}")
        return False

    tool_calls = (resp.additional_kwargs or {}).get("tool_calls") or []
    has_valid_tool_call = any(
        (tc.get("function") or {}).get("arguments") for tc in tool_calls
    )
    if verbose:
        print(
            f"[router] 探针：tool_calls={len(tool_calls)}, "
            f"has_valid_args={has_valid_tool_call}"
        )
    return has_valid_tool_call


# ---------------------------------------------------------------------------
# 2. 输出清洗：剥 <think> 标签 + 剥 markdown 围栏 + 提取 {...}
# ---------------------------------------------------------------------------
def strip_think_and_fence(text: str) -> str:
    """从模型 content 中提取纯 JSON 字符串。"""
    if not isinstance(text, str):
        text = getattr(text, "content", "") or ""
    # 1) 去掉 <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2) ```json ... ``` 围栏
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # 3) 兜底：找第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


# ---------------------------------------------------------------------------
# 3. 智能路由
# ---------------------------------------------------------------------------
class StructuredRouter:
    """
    智能路由：根据模型能力自动选择最佳结构化输出路线。

    用法：
        router = StructuredRouter(model)
        chain  = router.build(Person)           # 一次性探测 + 构建 chain
        person = chain.invoke("...")            # 后续直接 invoke 即可
    """

    def __init__(self, chat_model: BaseChatModel, verbose: bool = True):
        self.model = chat_model
        self.verbose = verbose
        self._route: str | None = None  # 缓存探测结果

    def _build_function_calling_chain(self, schema: Type[BaseModel]) -> Runnable:
        return self.model.with_structured_output(schema)

    def _build_prompt_parser_chain(self, schema: Type[BaseModel]) -> Runnable:
        parser = PydanticOutputParser(pydantic_object=schema)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是信息抽取助手。根据用户文本提取结构化字段。\n"
                    "严格要求：只输出合法 JSON，禁止任何思考过程、寒暄、解释或 markdown 围栏。\n"
                    "{format_instructions}",
                ),
                ("human", "{input}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        def clean(msg: Any) -> str:
            content = msg.content if isinstance(msg, AIMessage) else msg
            return strip_think_and_fence(content)

        return prompt | self.model | clean | parser

    def build(self, schema: Type[BaseModel], force: str | None = None) -> Runnable:
        """
        构建可用于 invoke 的 chain。

        Args:
            schema: Pydantic 模型类
            force : "function_calling" | "prompt_parser" | None
                    None 表示自动探测；否则强制使用某条路线（调试用）。
        """
        if force == "function_calling":
            self._route = "function_calling"
        elif force == "prompt_parser":
            self._route = "prompt_parser"
        else:
            self._route = (
                "function_calling"
                if detect_tool_calling_support(self.model, verbose=self.verbose)
                else "prompt_parser"
            )

        if self.verbose:
            print(f"[router] 选定路线: {self._route}")

        if self._route == "function_calling":
            return self._build_function_calling_chain(schema)
        return self._build_prompt_parser_chain(schema)

    @property
    def route(self) -> str | None:
        return self._route


# ---------------------------------------------------------------------------
# 4. 极简一行调用（可选）
# ---------------------------------------------------------------------------
def invoke_structured(
    model: BaseChatModel,
    schema: Type[BaseModel],
    text: str,
    **kwargs: Any,
) -> BaseModel:
    """极简入口：探测 + 构建 + invoke 三合一。"""
    chain = StructuredRouter(model, verbose=kwargs.pop("verbose", True)).build(schema)
    return chain.invoke(text, **kwargs)

