import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable
import networkx as nx

# ---------------------------------------------------------------------------
# 1. Agent Anatomy Formulation: A = Φ(L, P, C, E)
# ---------------------------------------------------------------------------

@dataclass
class AgentMemory:
    """Experience (E): 累積的互動歷史與上下文記憶"""
    history: List[Dict[str, str]] = field(default_factory=list)

    def add_record(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

class Agent:
    def __init__(self, agent_id: str, persona: str, model_name: str = "gpt-4o"):
        self.id = agent_id
        self.persona = persona  # Persona (P)
        self.model_name = model_name  # Cognitive Core (L)
        self.capabilities: Dict[str, Callable] = {}  # Capabilities (C)
        self.memory = AgentMemory()  # Experience (E)
        
        # 經濟與價值循環屬性 (Endogenous Value Cycle)
        self.value_score: float = 1.0
        self.reputation: float = 0.5
        
        # 註冊預設通用工具 (Base Agent Template)
        self.register_capability("system_log", self._default_log)

    def register_capability(self, name: str, func: Callable):
        """動態載入外部能力/工具"""
        self.capabilities[name] = func

    def _default_log(self, message: str) -> str:
        return f"[Agent {self.id} Log]: {message}"

    async def execute_task(self, task_description: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """基於 ReAct (Φ) 框架的異步任務執行"""
        self.memory.add_record("user", task_description)
        
        # 模擬 LLM 思考與呼叫 Capabilities
        print(f"🤖 [Agent {self.id}] 正在處理任務: '{task_description}'")
        await asyncio.sleep(0.5)  # 模擬推理延遲
        
        # 如果存在匹配能力則呼叫
        output_data = f"Processed '{task_description}' with inputs {inputs}"
        if "data_processing" in self.capabilities and "data" in inputs:
            output_data = self.capabilities["data_processing"](inputs["data"])

        self.memory.add_record("assistant", str(output_data))
        return {"status": "success", "result": output_data, "agent_id": self.id}


# ---------------------------------------------------------------------------
# 2. Nuwa Engine: Agent 的生成、維護與生命週期管理
# ---------------------------------------------------------------------------

class NuwaEngine:
    """女媧引擎：動態生成與管理高異質性 Agent"""
    def __init__(self):
        self.registry: Dict[str, Agent] = {}

    def spawn_agent(self, persona: str, capabilities: Dict[str, Callable] = None) -> Agent:
        agent_id = f"agent_{uuid.uuid4().hex[:6]}"
        agent = Agent(agent_id=agent_id, persona=persona)
        
        if capabilities:
            for name, func in capabilities.items():
                agent.register_capability(name, func)
                
        self.registry[agent_id] = agent
        print(f"✨ [NuwaEngine] 成功生成 Agent: {agent_id} (Role: {persona})")
        return agent

    def get_agent(self, agent_id: str) -> Agent:
        return self.registry.get(agent_id)


# ---------------------------------------------------------------------------
# 3. Market-Driven Orchestrator & DAG Verification
# ---------------------------------------------------------------------------

@dataclass
class TaskNode:
    node_id: str
    description: str
    required_capability: str
    dependencies: List[str] = field(default_factory=list)

class MarketOrchestrator:
    """市場驅動協調器：DAG 分析、拓樸排序與價值競價指派"""
    def __init__(self, nuwa_engine: NuwaEngine):
        self.nuwa = nuwa_engine

    def validate_and_sort_dag(self, tasks: List[TaskNode]) -> List[str]:
        """DAG 可達性與拓樸排序Guardrail"""
        graph = nx.DiGraph()
        for task in tasks:
            graph.add_node(task.node_id)
            for dep in task.dependencies:
                graph.add_edge(dep, task.node_id)
        
        # 檢查是否存在環狀依賴 (Cycle Check)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("❌ [Orchestrator Guardrail Check Failed]: 任務依賴關係包含環 (Cycle)！")
        
        # 返回拓樸排序後的執行順序
        return list(nx.topological_sort(graph))

    def auction_task(self, required_capability: str) -> Agent:
        """市場競價機制 (Endogenous Value Cycle)：選擇最適 Agent"""
        candidates = [
            agent for agent in self.nuwa.registry.values()
            if required_capability in agent.capabilities
        ]
        
        if not candidates:
            # 若無特化 Agent，回傳任一 Base Agent
            return list(self.nuwa.registry.values())[0]
        
        # 依據 Value Score 與 Reputation 進行選擇
        best_agent = max(candidates, key=lambda a: a.value_score * 0.6 + a.reputation * 0.4)
        return best_agent

    async def dispatch_pipeline(self, task_nodes: List[TaskNode]):
        """非同步管道排程與執行"""
        execution_order = self.validate_and_sort_dag(task_nodes)
        task_map = {t.node_id: t for t in task_nodes}
        results: Dict[str, Any] = {}

        print(f"\n🚀 [Orchestrator] 任務 Pipeline 通過 DAG 驗證，最佳執行順序: {execution_order}\n")

        for node_id in execution_order:
            node = task_map[node_id]
            
            # 收集前置任務依賴的輸出
            input_data = {dep: results[dep]["result"] for dep in node.dependencies if dep in results}
            
            # 市場指派 Agent
            assigned_agent = self.auction_task(node.required_capability)
            
            # 執行任務
            res = await assigned_agent.execute_task(node.description, input_data)
            results[node_id] = res
            
            # 價值回饋與進化 (Value Feedback Loop)
            assigned_agent.value_score += 0.1
            assigned_agent.reputation += 0.05

        return results


# ---------------------------------------------------------------------------
# 4. 系統運行示範 (Demo Run)
# ---------------------------------------------------------------------------

async def main():
    # 1. 初始化 Nuwa 引擎並動態生成 Agent 生態系
    nuwa = NuwaEngine()
    
    # 特化 Capability 函數
    def fetch_web_data(inputs):
        return {"raw_text": "Agentic Web paper data content..."}

    def analyze_data(inputs):
        return f"Summarized: {inputs.get('task_1', {}).get('raw_text', '')}"

    # 生態系生成不同角色的 Agent
    nuwa.spawn_agent(persona="Web Crawler", capabilities={"data_fetching": fetch_web_data})
    nuwa.spawn_agent(persona="Data Analyst", capabilities={"data_processing": analyze_data})

    # 2. 定義 Web-Scale 多階段任務 (DAG 結構)
    tasks = [
        TaskNode(
            node_id="task_1",
            description="抓取網頁資料",
            required_capability="data_fetching"
        ),
        TaskNode(
            node_id="task_2",
            description="分析抓取到的數據",
            required_capability="data_processing",
            dependencies=["task_1"]
        ),
    ]

    # 3. 啟動 Orchestrator 調度執行
    orchestrator = MarketOrchestrator(nuwa_engine=nuwa)
    final_results = await orchestrator.dispatch_pipeline(tasks)

    print("\n✅ [Pipeline 執行完畢結果]:")
    print(json.dumps(final_results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())

