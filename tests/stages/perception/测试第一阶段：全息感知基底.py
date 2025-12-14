import asyncio

from src.state_of_mind.core.orchestration import MetaCognitiveOrchestrator

orchestrator = MetaCognitiveOrchestrator()
text = """
夜晚九点，夏末的微风轻拂。客厅里，月光透过纱帘洒在木地板上，像一层薄霜。母亲坐在旧藤椅上，穿着洗得发软的白色短袖，头发随意挽起，几缕碎发贴在额角。她望着站在阳台边的女儿，声音微微发颤：
“我为你牺牲了事业，放弃再婚机会，你就不能留在本地陪我吗？”
女儿没有回头，手指无意识地摩挲着玻璃门框。她目光凝在远处楼宇间的一线夜空，睫毛低垂，嘴唇微动，却终究没说出话——一种沉重的无奈压在胸口，像被无形的手攥住。
片刻沉默后，母亲的声音再次响起，更低，更沉，带着一丝哽咽：
“你要是走了，别人会说我不孝，我这半辈子白养你了。”
"""


async def analyze_text():
    try:
        result = await orchestrator.run(stage_name="perception", user_input=text)
        print(result)
    except Exception as e:
        raise


if __name__ == "__main__":
    asyncio.run(analyze_text())
