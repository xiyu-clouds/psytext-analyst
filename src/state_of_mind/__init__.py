from src.state_of_mind.core.llm.deepseek import AsyncDeepSeekBackend
from src.state_of_mind.core.llm.qwen import AsyncQwenLLMBackend
from src.state_of_mind.utils.registry import GlobalSingletonRegistry
from src.state_of_mind.utils.constants import ModelName

GlobalSingletonRegistry.register_backend(ModelName.QWEN, AsyncQwenLLMBackend)
GlobalSingletonRegistry.register_backend(ModelName.DEEPSEEK, AsyncDeepSeekBackend)
