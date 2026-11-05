"""End-to-end A2A test: two remote agents over localhost HTTP, signed lineage, async polling."""
import asyncio, time, holos, a2a

KEY = b"platform-secret"
def writer(text, meta):
    time.sleep(0.2)                                   # long-ish task -> client polls tasks/get
    return "FINAL: report drafted remotely. [quality=0.93]"
def analyst(text, meta): return "FINAL: analysis done. [quality=0.88]"

s1 = a2a.A2AServer("alice-writer", [("report-writing", "technical report writing", ["writing"])], writer, price=2.0, token="tok", verify_key=KEY)
s2 = a2a.A2AServer("alice-analyst", [("analysis", "data analysis of financial figures", ["data"])], analyst, price=1.5, token="tok", verify_key=KEY)

# transport-level checks
bad = a2a.A2AClient(token="wrong"); good = a2a.A2AClient(token="tok", signing_key=KEY)
for name, fn in [("bad token", lambda: bad.discover(s1.base) and bad.rpc(s1.base + "/a2a", "message/send", {})),
                 ("unsigned message", lambda: a2a.A2AClient(token="tok").send(s1.base + "/a2a", "hi", ctx=None)),
                 ("unknown method", lambda: good.rpc(s1.base + "/a2a", "foo/bar", {}))]:
    try: fn(); print(f"{name}: UNEXPECTEDLY OK")
    except a2a.A2AError as e: print(f"{name}: rejected -> {e}")
print("card:", good.discover(s1.base)["name"], good.discover(s1.base)["skills"][0]["name"])

async def main():
    h = holos.Holos(n_agents=1500, seed=1)
    await a2a.probe_and_register(h, "alice", [s1.base, s2.base, "http://127.0.0.1:9"], good)
    r = await h.run("edge-AI chip market analysis", budget=40, constraints={"user_id": "alice"})
    print({k: v for k, v in r["nodes"].items()}, "chain_valid:", r["chain_valid"])
    print("server-side lineage seen by alice-writer:", s1.received)
    print("ledger:", {k: round(v["free"] + v["locked"], 2) for k, v in h.ledger.acc.items()})
asyncio.run(main())
