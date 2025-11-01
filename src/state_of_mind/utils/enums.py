# =============== 核心模型类（类型安全 + 显式配置） ===============
from enum import Enum
from typing import List, Dict

from src.state_of_mind.utils.constants import ModelCapability, ModelName, MODEL_CONFIG


class Model(str, Enum):
    """
    统一模型枚举类，支持跨厂商调度。
    所有能力信息来自官方文档或可信配置，而非字符串推测。
    """

    # === Qwen Models ===
    QWEN_MAX = ModelName.QWEN_MAX
    QWEN3_MAX = ModelName.QWEN3_MAX
    QWEN_PLUS = ModelName.QWEN_PLUS
    QWEN_FLASH = ModelName.QWEN_FLASH

    # === DeepSeek Models ===
    DEEPSEEK_REASONER = ModelName.DEEPSEEK_REASONER
    DEEPSEEK_CHAT = ModelName.DEEPSEEK_CHAT

    def __new__(cls, value: str):
        # 创建实例并绑定 _value_
        obj = str.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value: str):
        super().__init__()
        config = MODEL_CONFIG.get(value)
        if not config:
            raise ValueError(f"Missing configuration for model: {value}")
        self.capabilities: Dict[str, bool] = config["capabilities"]
        self.provider: str = config["provider"]

    # === 属性访问（IDE 可识别）===
    @property
    def value(self) -> str:
        return str(self)

    # === 能力查询方法 ===
    def supports_json_format(self) -> bool:
        return self.capabilities.get(ModelCapability.JSON_FORMAT, False)

    def is_reasoning_model(self) -> bool:
        return self.capabilities.get(ModelCapability.REASONING, False)

    def is_coder_model(self) -> bool:
        return self.capabilities.get(ModelCapability.CODE, False)

    def is_vision_model(self) -> bool:
        return self.capabilities.get(ModelCapability.VISION, False)

    def is_audio_model(self) -> bool:
        return self.capabilities.get(ModelCapability.AUDIO, False)

    def is_emotion_model(self) -> bool:
        return self.capabilities.get(ModelCapability.EMOTION, False)

    def supports_streaming(self) -> bool:
        return self.capabilities.get(ModelCapability.STREAMING, True)

    # === 元信息 ===
    @property
    def name(self) -> str:
        return self.value

    # === 工具方法 ===
    @classmethod
    def has_value(cls, value: str) -> bool:
        try:
            cls(value)
            return True
        except ValueError:
            return False

    @classmethod
    def list_all(cls) -> List['Model']:
        return list(cls.__members__.values())

    @classmethod
    def list_by_provider(cls, provider: str) -> List['Model']:
        return [m for m in cls.list_all() if m.provider == provider]

    @classmethod
    def list_by_capability(cls, capability: str) -> List['Model']:
        return [m for m in cls.list_all() if m.capabilities.get(capability, False)]

    def __repr__(self) -> str:
        return f"<Model.{self.name} provider={self.provider} capabilities={self.capabilities}>"
