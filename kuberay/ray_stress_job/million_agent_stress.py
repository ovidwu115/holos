import argparse
import time
import ray


@ray.remote
class SimulatedAgent:
    def __init__(self, agent_id: int, state_bytes: int = 256):
        self.agent_id = agent_id
        self.state = bytearray(state_bytes)
        self.ticks = 0

    def ready(self) -> int:
        return self.agent_id

    def step(self, payload_size: int = 64) -> dict:
        self.ticks += 1
        _ = sum(self.state[: min(len(self.state), 32)]) + payload_size
        return {
            "agent_id": self.agent_id,
            "ticks": self.ticks,
            "node_ip": ray.util.get_node_ip_address(),
        }


def run_stress_test(target_agents: int, batch_size: int, step_rounds: int, state_bytes: int):
    ray.init(ignore_reinit_error=True)
    cluster_res = ray.cluster_resources()
    print(f"[Init] Connected to Ray cluster. Resources: {cluster_res}")

    t0 = time.perf_counter()
    actors = []

    for start in range(0, target_agents, batch_size):
        end = min(start + batch_size, target_agents)
        b_start = time.perf_counter()
        batch = [
            SimulatedAgent.remote(agent_id=i, state_bytes=state_bytes)
            for i in range(start, end)
        ]
        # 透過呼叫 ready() 取得 ObjectRef，等待本批 Actor 真正建立完成
        ray.get([a.ready.remote() for a in batch])
        actors.extend(batch)
        elapsed = time.perf_counter() - b_start
        print(
            f"[Spawn] Created {len(actors)}/{target_agents} agents "
            f"(batch={len(batch)} in {elapsed:.2f}s)"
        )

    spawn_time = time.perf_counter() - t0
    print(f"[Summary] Spawned {len(actors)} agents in {spawn_time:.2f}s")

    for r in range(1, step_rounds + 1):
        r_start = time.perf_counter()
        futures = [a.step.remote() for a in actors]
        results = ray.get(futures)
        r_elapsed = time.perf_counter() - r_start
        rps = len(results) / max(r_elapsed, 1e-6)
        print(
            f"[Round {r}/{step_rounds}] Completed {len(results)} steps in "
            f"{r_elapsed:.2f}s ({rps:,.0f} steps/s)"
        )

    total_time = time.perf_counter() - t0
    print(f"[Done] Total stress test time: {total_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Ray Million-Agent Stress Test")
    parser.add_argument("--target-agents", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--step-rounds", type=int, default=3)
    parser.add_argument("--state-bytes", type=int, default=256)
    args = parser.parse_args()

    run_stress_test(
        target_agents=args.target_agents,
        batch_size=args.batch_size,
        step_rounds=args.step_rounds,
        state_bytes=args.state_bytes,
    )


if __name__ == "__main__":
    main()

