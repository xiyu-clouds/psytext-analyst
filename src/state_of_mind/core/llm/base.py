from typing import Dict, Any
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """
    抽象基类：所有 LLM 后端必须继承
    实现 async_call / generate_text / close
    """
    CHINESE_NAME = "抽象基类LLM后端"

    def __init__(self):
        self.initialized = False
        self.configs = {}  # 存储连接级配置（如 api_key, timeout）

    async def init(self, configs: dict = None):
        """
        异步初始化，子类可重写
        config: 仅包含影响连接行为的参数（如 api_key, timeout）
        """
        self.configs = configs or {}
        self.initialized = True
        return self

    @abstractmethod
    async def async_call(self, prompt: str, model: str, category: str, params: dict) -> Dict[str, Any]:
        """异步调用，返回标准结构"""
        pass

    @abstractmethod
    async def generate_text(self, prompt: str, model: str, params: dict) -> str:
        """异步生成纯文本"""
        pass

    @abstractmethod
    async def bottom_dissolving_pronouns(self, prompt: str, model: str, params: dict) -> Dict[int, str]:
        """异步兜底消解代词指称，返回 {index: name} 映射，失败时返回空 dict"""
        pass

    @abstractmethod
    async def close(self):
        """关闭客户端资源（必须显式调用）"""
        pass
