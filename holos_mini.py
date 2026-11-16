import asyncio, hashlib, json, random
from dataclasses import dataclass

async def mock_llm(prompt: str) -> str:
    await asyncio.sleep(0.01)
    return f"[mock] {prompt[:48]} -> done"

@dataclass
class Agent:
    id: str; rqt: dict; credit: float = 100.0
    async def to_dormant(self, ledger):
        summary = await mock_llm(f"summarize {self.id}")
        ledger.record(self.id, {"summary": summary, "credit": self.credit})
        return {"id_hash": hashlib.sha256(self.id.encode()).hexdigest()[:8]}

class NuwaEngine:
    async def instantiate(self, seed_id: str) -> Agent:
        await asyncio.sleep(0.02)  # 模擬冷啟動
        return Agent(id=seed_id, rqt={"role": "worker", "task": "demo"})

class Dispatcher:
    async def dispatch(self, task: str, agents: list[Agent]):
        ranked = sorted(agents, key=lambda a: a.credit, reverse=True)
        return ranked[0] if ranked else None

class Orchestrator:
    async def run(self, task: str):
        engine, dispatcher = NuwaEngine(), Dispatcher()
        agent = await engine.instantiate("agent-001")
        winner = await dispatcher.dispatch(task, [agent])
        result = await mock_llm(f"{winner.id} executes: {task}")
        return {"winner": winner.id, "result": result}

async def main():
    print(await Orchestrator().run("summarize 100 docs"))

asyncio.run(main())
