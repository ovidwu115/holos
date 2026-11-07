import ray
import time
import random
import asyncio

# 初始化 Ray 叢集（可連線至雲端 Kubernetes Ray Cluster）
# ray.init(address="auto") # 實際分散式環境使用
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 1. 分散式 Agent Actor (利用 Ray Actor 實現百萬級水平擴展)
# ---------------------------------------------------------------------------
@ray.remote
class ScalableAgentActor:
    def __init__(self, agent_id: str, persona: str):
        self.agent_id = agent_id
        self.persona = persona
        self.value_score = 1.0
        self.reputation = 0.5
        self.memory_cache = []

    async def handle_task(self, task_payload: dict) -> dict:
        """模擬 Agent 接收並處理任務（包含輕量推理與狀態更新）"""
        # 模擬運算延遲 (模擬 LLM 推理或工具調用)
        await asyncio.sleep(random.uniform(0.01, 0.05))
        
        # 狀態變更
        self.value_score += 0.01
        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "updated_value": self.value_score
        }

    def get_status(self):
        return {"id": self.agent_id, "value_score": self.value_score}


# ---------------------------------------------------------------------------
# 2. 分散式壓測協調器 (Load Generator & Orchestrator)
# ---------------------------------------------------------------------------
class MillionAgentStressTester:
    def __init__(self, total_agents: int = 10000): # 示範規模，可拉高至 1,000,000
        self.total_agents = total_agents
        self.agents = []

    def setup_cluster(self):
        print(f"🚀 [StressTester] 正在初始化 {self.total_agents} 個分散式 Agent...")
        start_time = time.time()
        
        # 批次生成 Ray Actors
        self.agents = [
            ScalableAgentActor.remote(agent_id=f"agent_{i}", persona="WorkerBot")
            for i in range(self.total_agents)
        ]
        print(f"✨ 成功部署完成，耗時: {time.time() - start_time:.2f} 秒")

    async def run_load_test(self, concurrent_tasks: int = 5000):
        print(f"\n🔥 [StressTester] 開始執行壓力測試：發送總計 {concurrent_tasks} 個並發任務...")
        start_time = time.time()

        # 隨機挑選 Agent 模擬市場任務分派
        tasks = []
        for _ in range(concurrent_tasks):
            selected_agent = random.choice(self.agents)
            task_payload = {"instruction": "Process web-scale query", "data": random.randint(1, 100)}
            # 非同步呼叫遠端 Ray Actor
            tasks.append(selected_agent.handle_task.remote(task_payload))

        # 等待所有非同步任務完成
        results = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        tps = concurrent_tasks / duration
        print(f"📊 [壓測報告]")
        print(f"   - 成功處理任務數: {len(results)}")
        print(f"   - 總耗時: {duration:.4f} 秒")
        print(f"   - 系統吞吐量 (Throughput): {tps:.2f} 任務/秒")

# ---------------------------------------------------------------------------
# 3. 執行壓測
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tester = MillionAgentStressTester(total_agents=10000) # 測試時可按硬體擴充
    tester.setup_cluster()
    
    # 執行非同步壓測迴圈
    asyncio.run(tester.run_load_test(concurrent_tasks=20000))

