import traceback
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class LLMResponse(BaseModel):
    """
    统一的 LLM 调用响应结构。
    核心状态由两个布尔字段决定：
      - __success: 是否成功完成一次 LLM 调用（含收到有效或错误响应）
      - __valid_structure: 返回内容是否符合预期 JSON 结构
    """
    model_config = ConfigDict(populate_by_name=True)

    # === 核心状态（必须显式传入）===
    success: bool = Field(alias="__success")
    valid_structure: bool = Field(alias="__valid_structure")

    # === 数据与错误信息 ===
    data: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Optional[str] = Field(default=None, alias="__raw_response")
    validation_errors: List[str] = Field(default_factory=list, alias="__validation_errors")
    api_error: Optional[str] = Field(default=None, alias="__api_error")
    system_error: Optional[str] = Field(default=None, alias="__system_error")
    traceback: Optional[str] = Field(default=None, alias="__traceback")

    # === 上下文元数据 ===
    model: str = Field(default="unknown")
    template_name: str = Field(default="")
    step_name: str = Field(default="")
    prompt_type: str = Field(default="")

    # === 性能指标（可选）===
    usage: Optional[Dict[str, Any]] = Field(default=None)
    latency_ms: Optional[float] = Field(default=None)

    def is_success(self) -> bool:
        """业务成功 = 接口调用成功 + 结构有效"""
        return self.__success and self.__valid_structure

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)

    # ==================== 工厂方法 ====================

    @classmethod
    def from_successful_call(
            cls,
            *,
            valid_structure: bool,
            data: Dict[str, Any],
            raw_response: Optional[str] = None,
            validation_errors: Optional[List[str]] = None,
            api_error: Optional[str] = None,
            model: str,
            template_name: str,
            step_name: str,
            prompt_type: str,
            latency_ms: Optional[float] = None,
            usage: Optional[Dict[str, Any]] = None,
    ) -> "LLMResponse":
        """用于 HTTP 200 响应（无论内容是否有效）"""
        return cls(
            success=True,
            valid_structure=valid_structure,
            data=data or {},
            raw_response=raw_response,
            validation_errors=validation_errors or [],
            api_error=api_error,
            system_error=None,
            traceback=None,
            model=model,
            template_name=template_name,
            step_name=step_name,
            prompt_type=prompt_type,
            latency_ms=latency_ms,
            usage=usage,
        )

    @classmethod
    def from_api_error(
            cls,
            *,
            status_code: int,
            error_message: str,
            model: str,
            template_name: str,
            step_name: str,
            prompt_type: str,
            latency_ms: Optional[float] = None,
    ) -> "LLMResponse":
        """用于 HTTP 非 200 响应（如 401, 400, 429, 500 等），但请求已发出并收到响应"""
        return cls(
            success=False,
            valid_structure=False,
            data={},
            raw_response=None,
            validation_errors=[],
            api_error=f"[{status_code}] {error_message}",
            system_error=None,
            traceback=None,
            model=model,
            template_name=template_name,
            step_name=step_name,
            prompt_type=prompt_type,
            latency_ms=latency_ms,
        )

    @classmethod
    def from_system_error(
            cls,
            *,
            system_error: str,
            model: str = "unknown",
            template_name: str = "",
            step_name: str = "",
            prompt_type: str = "",
            include_traceback: bool = True,
    ) -> "LLMResponse":
        """用于请求未完成的系统级异常（网络、超时、代码崩溃等）"""
        tb = traceback.format_exc() if include_traceback else None
        return cls(
            success=False,
            valid_structure=False,
            data={},
            raw_response=None,
            validation_errors=[],
            api_error=None,
            system_error=system_error,
            traceback=tb,
            model=model,
            template_name=template_name,
            step_name=step_name,
            prompt_type=prompt_type,
        )
