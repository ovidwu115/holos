# 論文（arXiv 2604.02334）實作計畫

 holos.py，純 Python + numpy，離線可跑。

## Installation 
`git clone https://github.com/ovidwu115/holos`
`pip install numpy`


## Usage 

跑完整流程，

`python holos.py demo`

跑經濟模擬。

`python holos.py sim`

跑 bench

`python bench_nuwa.py`

```
  seeds    50,000     0.3s
  seeds   100,000     0.6s
  seeds   149,999     0.9s
  seeds   199,997     1.3s
  seeds   249,995     1.7s
  seeds   299,992     2.1s
  seeds   349,991     2.6s
  seeds   399,983     3.0s
  seeds   449,978     3.4s
  seeds   499,973     3.8s
  seeds   549,964     4.2s
  seeds   599,958     4.7s
  seeds   649,952     5.1s
  seeds   699,937     5.6s
  seeds   749,934     6.1s
  seeds   799,927     6.5s
  seeds   849,916     7.0s
  seeds   899,907     7.5s
  seeds   949,897     8.0s
  seeds   999,880     8.5s
  seeds 1,000,000     8.6s

N=1,000,000  create 8.6s (116,819 seeds/s, no LLM)
disk: 193.4 MB  (193 B/agent; 10M est 1.93 GB)
RSS: 1338 MB total (vector index 1024 MB = 1024 B/agent)

population | point lookup mean/P99 (ms) | top-10 retrieval mean/P99 (ms)
        10 |   0.009 /   0.022        |     0.01 /     0.03
       100 |   0.009 /   0.018        |     0.01 /     0.02
     1,000 |   0.012 /   0.025        |     0.04 /     0.06
    10,000 |   0.011 /   0.018        |     0.46 /     0.82
   100,000 |   0.013 /   0.023        |     7.34 /     7.78
 1,000,000 |   0.014 /   0.021        |    74.96 /    84.18

JIT instantiate (profile fetch + S-MMR K=5, 40-tool pool): mean 0.31 ms, P99 0.38 ms
final RSS 1352 MB
```
跑 Gemini 版本

`python holos_gemini.py`

```
✨ [NuwaEngine] 成功生成 Agent: agent_e7a2c8 (Role: Web Crawler)
✨ [NuwaEngine] 成功生成 Agent: agent_63faab (Role: Data Analyst)

🚀 [Orchestrator] 任務 Pipeline 通過 DAG 驗證，最佳執行 順序: ['task_1', 'task_2']

🤖 [Agent agent_e7a2c8] 正在處理任務: '抓取網頁資料'
🤖 [Agent agent_63faab] 正在處理任務: '分析抓取到的數據'

✅ [Pipeline 執行完畢結果]:
{
  "task_1": {
    "status": "success",
    "result": "Processed '抓取網頁資料' with inputs {}",
    "agent_id": "agent_e7a2c8"
  },
  "task_2": {
    "status": "success",
    "result": "Processed '分析抓取到的數據' with inputs {'task_1': \"Processed '抓取網頁資料' with inputs {}\"}",
    "agent_id": "agent_63faab"
  }
}
```

跑 Gemini 百萬壓測

`
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
`
```
(raylet) WARNING: 82 PYTHON worker processes have been started on node
```
ray.exceptions.OutOfMemoryError: