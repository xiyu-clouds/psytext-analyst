from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import hashlib
import json
from src.state_of_mind.utils.logger import LoggerManager as logger


class BaseCache(ABC):
    CHINESE_NAME = "抽象基类缓存器"

    @staticmethod
    def make_key(template_name: str, **template_vars) -> str:
        try:
            normalized_vars = {}
            for k, v in template_vars.items():
                if v is None:
                    v = "None"
                elif isinstance(v, bool):
                    v = str(v).lower()
                elif isinstance(v, (int, float)):
                    if isinstance(v, float):
                        v = round(v, 10)
                elif isinstance(v, (list, dict)):
                    v = json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
                elif not isinstance(v, str):
                    v = str(v)
                normalized_vars[k] = v

            sorted_items = tuple(sorted(normalized_vars.items()))
            repr_str = f"{template_name}||{sorted_items}"
            return hashlib.md5(repr_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"缓存 key 生成失败，使用 fallback: {e}")
            return hashlib.md5(template_name.encode('utf-8')).hexdigest()

    # ========== 异步底层存储接口（子类必须实现）==========
    @abstractmethod
    async def _aget_raw(self, key: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def _aset_raw(self, key: str, value: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def _adelete_raw(self, key: str) -> None:
        pass

    @abstractmethod
    async def _aclear_raw(self) -> None:
        pass

    async def _akeys_raw(self) -> List[str]:
        raise NotImplementedError

    # ========== 统一高层异步接口（子类无需重写）==========
    async def get(self, key: str) -> Dict[str, Any]:
        try:
            data = await self._aget_raw(key)
            return {"success": True, "data": data, "error": None}
        except Exception as e:
            logger.error(f"Cache get failed for {key}: {e}")
            return {"success": False, "data": None, "error": str(e)}

    async def set(self, key: str, value: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await self._aset_raw(key, value)
            return {"success": True, "data": None, "error": None}
        except Exception as e:
            logger.error(f"Cache set failed for {key}: {e}")
            return {"success": False, "data": None, "error": str(e)}

    async def delete(self, key: str) -> Dict[str, Any]:
        try:
            await self._adelete_raw(key)
            return {"success": True, "data": None, "error": None}
        except Exception as e:
            logger.error(f"Cache delete failed for {key}: {e}")
            return {"success": False, "data": None, "error": str(e)}

    async def clear(self) -> Dict[str, Any]:
        try:
            await self._aclear_raw()
            return {"success": True, "data": None, "error": None}
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return {"success": False, "data": None, "error": str(e)}
