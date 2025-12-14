from typing import Protocol, Dict, Any, List


class StageProtocol(Protocol):
    """
    三阶段（Perception / Abstraction / Ontology）及 Feedback 的最小执行协议。
    """

    async def run(self, user_input: str, category: str, **kwargs) -> Dict[str, Any]:
        """
        执行单次推理。
        """
        pass

    async def run_batch(self, inputs: List[str], category: str, **kwargs) -> List[Dict[str, Any]]:
        """
        批量执行。保持输入输出顺序一致。
        """
        pass
