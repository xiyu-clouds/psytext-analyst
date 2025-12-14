from typing import Dict, Any, List
from src.state_of_mind.stages.perception.stage_pipeline import PerceptionPipeline


class MetaCognitiveOrchestrator:
    def __init__(self):
        self.stages: Dict[str, Any] = {
            "perception": PerceptionPipeline(),
            # "abstraction": AbstractionStage(),
            # "ontology": OntologyStage(),
            # "feedback": FeedbackStage(),
        }

    async def run(self, stage_name: str, user_input: str, **kwargs) -> Dict[str, Any]:
        if stage_name not in self.stages:
            available = list(self.stages.keys())
            raise ValueError(f"未知阶段 '{stage_name}'，可用阶段: {available}")

        stage = self.stages[stage_name]
        return await stage.run(user_input, **kwargs)

    async def run_batch(self, stage_name: str, user_inputs: List[str], **kwargs) -> List[Dict[str, Any]]:
        if stage_name not in self.stages:
            raise ValueError(f"未知阶段: {stage_name}")
        return await self.stages[stage_name].run_batch(user_inputs, **kwargs)
