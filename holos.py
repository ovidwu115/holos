#!/usr/bin/env python3
"""
Holos reference implementation (arXiv 2604.02334, "Holos: A Web-Scale LLM-Based
Multi-Agent System for the Agentic Web").

Layers (paper section):
  Interface  3.3  AgentCard / RemoteAgent / private channels (A2A-style, minimal)
  Substrate  3.4  Nuwa: RQT genotype -> dormant seeds (32-bit ids) -> JIT agents, S-MMR tools
  Coordination 3.5 Orchestrator (blind planning + Kahn validation + Continue/Retry/Modify),
                  Dispatcher (hybrid sourcing + learning-to-rank), persistent work loop
  Value      3.6  Ledger (7-phase cycle), AgentRank (signed), mandate hash chain
  Outcome    3.7  Evaluator (0-10, independent, regex-parsed), artifact store

LLMs are pluggable: llm(kind, prompt, **meta) -> str.  MockLLM runs fully offline.
Usage:  python holos.py demo | python holos.py sim
"""
from __future__ import annotations
import asyncio, copy, hashlib, hmac, json, random, re, sqlite3, sys, time, zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import numpy as np

# ───────────────────────── utilities ─────────────────────────
DIM = 256

def embed(text: str) -> np.ndarray:
    """Signed feature hashing; swap for a real embedding model in production."""
    v = np.zeros(DIM)
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 8) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v

def minmax(x):
    x = np.asarray(x, float); r = x.max() - x.min()
    return (x - x.min()) / r if r > 1e-9 else np.zeros_like(x)

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

# ───────────────────────── mock LLM ─────────────────────────
class MockLLM:
    """Deterministic stand-in so the whole system runs offline."""
    TIER_BONUS = {"small": -0.05, "medium": 0.0, "frontier": 0.05}
    def __init__(self, seed=0, cyclic_first_plan=True):
        self.rng = random.Random(seed); self.cyc = cyclic_first_plan; self.plans = 0

    def __call__(self, kind, prompt, **m):
        if kind == "planner":
            self.plans += 1
            nodes = [
                dict(id="n1", desc=f"Collect sources and key facts on: {prompt}", skill="web research fact collection", out="facts.md"),
                dict(id="n2", desc="Analyze collected facts and extract trends", skill="data analysis", out="analysis.md", deps=["n1"]),
                dict(id="n3", desc="Create charts of the key figures", skill="data visualization charting", out="charts.md", deps=["n1"]),
                dict(id="n4", desc="Write the final report", skill="technical report writing", out="report.md", deps=["n2", "n3"]),
            ]
            if self.cyc and self.plans == 1:          # simulate an LLM-hallucinated cycle
                nodes[0]["deps"] = ["n4"]
            return json.dumps({"nodes": nodes})
        if kind == "profile":
            flavor = self.rng.choice(["methodical", "fast", "thorough", "pragmatic", "curious"])
            return f"{m['role']} ({m['qual']}) specialized in {m['task']}. Working style: {flavor}."
        if kind == "directive":
            return f"As [{m['profile'][:70]}], produce {m.get('io','output')}: {prompt}"
        if kind == "worker":
            a = m["agent"]
            q = min(1, max(0, a.ability + self.rng.gauss(0, 0.08) + self.TIER_BONUS.get(a.tier, 0)))
            return f"FINAL: result using {len(a.tools)} tools. [quality={q:.2f}]"
        if kind == "evaluator":
            q = float(re.search(r"quality=([\d.]+)", prompt).group(1))
            return f"Audit: completeness/accuracy/utility checked.\nSCORE: {10*q:.1f}"
        if kind == "decide":
            s, att, mx, n = m["score"], m["attempts"], m["max_attempts"], m["node"]
            if att == 1 and s >= 4: return json.dumps({"action": "retry", "strategy": "refine"})
            if att < mx: return json.dumps({"action": "retry", "strategy": "redispatch"})
            if not n["modified"]:
                nn = dict(id=n["id"] + "m", desc="Simplified: " + n["desc"], skill=n["skill"], out=n["out"],
                          deps=n["deps"], const=n["const"], modified=True)
                return json.dumps({"action": "modify", "nodes": [nn]})
            return json.dumps({"action": "fallback"})
        raise ValueError(kind)

# ───────────────────────── Interface layer ─────────────────────────
@dataclass
class AgentCard:                      # A2A-style metadata profile
    name: str; url: str; skills: list; provider: str = "user"

# ───────────────────────── Agents ─────────────────────────
@dataclass
class Agent:
    """A = Phi(L, P, C, E): LLM tier, persona, tools, experience; Phi = ReAct loop."""
    id: object; profile: str; tier: str; tools: list; ability: float; llm: object
    memory: list = field(default_factory=list)
    price: float = 1.0
    def run(self, directive, max_steps=3):
        scratch = ""
        for _ in range(max_steps):    # reason -> act -> reflect (tool execution omitted)
            out = self.llm("worker", directive + scratch, agent=self)
            if "FINAL:" in out: break
            scratch += "\n" + out
        self.memory.append(out[:200]); return out

class RemoteAgent(Agent):
    """User-owned agent living outside the platform, reached via a 'green lane'."""
    def __init__(self, card: AgentCard, handler, price=1.0):
        super().__init__(f"remote:{card.name}", " ".join(card.skills), "remote", [], 0.0, None, price=price)
        self.card, self.handler, self.ctx = card, handler, {}
    def run(self, directive, max_steps=3): return self.handler(directive)

# ───────────────────────── Substrate layer: Nuwa ─────────────────────────
ROLES = ["Senior Analyst", "Research Associate", "Data Engineer", "Technical Writer", "Visual Designer", "Auditor"]
QUALS = ["privacy-focused", "detail-oriented", "fast-turnaround", "creative", "cost-sensitive", "rigorous"]
TASK_TOPIC = {
    "web research and fact collection": "research", "market trend analysis": "research",
    "data analysis of financial figures": "data", "SQL data engineering": "data",
    "data visualization and charting": "viz", "UI design": "viz",
    "technical report writing": "writing", "translation": "writing",
    "code review": "code", "legal contract review": "legal"}
TOPIC_KW = {"research": "web research sources facts", "data": "data analysis statistics sql",
            "viz": "data visualization charting design", "writing": "technical report writing",
            "code": "code review software", "legal": "legal contract"}
TIERS = ["small", "medium", "frontier"]; TIER_PRICE = {"small": 1.0, "medium": 2.0, "frontier": 4.0}
INCOMPATIBLE = {("Visual Designer", "SQL data engineering"), ("Auditor", "UI design"), ("Technical Writer", "code review")}

def default_tool_pool(per_domain=5):
    kw = {"web": "search fetch scrape sources", "data": "pandas sql csv statistics analysis", "viz": "plot chart visualization map",
          "writing": "markdown report docx draft", "code": "git lint test compile review", "legal": "contract clause law",
          "math": "solver symbolic numeric", "media": "image audio video"}
    return [(f"{d}_tool_{i}", f"{d} {words} variant{i}") for d, words in kw.items() for i in range(per_domain)]

def smmr(v_a, tool_vecs, K, sigma, lam, seed):
    """Algorithm 1: Stochastic Maximal Marginal Relevance tool selection."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, len(tool_vecs))
    rel = tool_vecs @ v_a + eps
    sim = tool_vecs @ tool_vecs.T
    S, cand = [], list(range(len(tool_vecs)))
    while len(S) < min(K, len(tool_vecs)):
        red = np.array([max(sim[i, j] for j in S) if S else 0.0 for i in cand])
        score = lam * rel[cand] - (1 - lam) * red
        best = cand[int(np.argmax(score))]; S.append(best); cand.remove(best)
    return S

class Nuwa:
    def __init__(self, llm, tool_pool=None, seed=0):
        self.llm, self.rng = llm, random.Random(seed)
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE seeds(id INTEGER PRIMARY KEY, role, qual, task, profile, ability REAL, price REAL)")
        self.tools = tool_pool or default_tool_pool()
        self.tool_vecs = np.stack([embed(d) for _, d in self.tools])
        self.ids = np.array([], dtype=np.int64); self.mat = np.zeros((0, DIM), np.float32)
        self.topics = defaultdict(list); self.pos = {}

    def synthesize(self, n, discriminator=None):
        """RQT genotype -> discriminator -> phenotype variants -> dormant seeds keyed by 32-bit hash."""
        ok = discriminator or (lambda r, q, t: (r, t) not in INCOMPATIBLE)
        made = 0
        while made < n:
            r, q, t = self.rng.choice(ROLES), self.rng.choice(QUALS), self.rng.choice(list(TASK_TOPIC))
            if not ok(r, q, t): continue
            k = self.rng.randrange(10**6)
            aid = zlib.crc32(f"{r}|{q}|{t}|{k}".encode())
            tier = TIERS[aid % len(TIERS)]
            prof = self.llm("profile", "", role=r, qual=q, task=t)
            cur = self.db.execute("INSERT OR IGNORE INTO seeds VALUES(?,?,?,?,?,?,?)",
                                  (aid, r, q, t, prof, self.rng.uniform(0.3, 0.95), TIER_PRICE[tier]))
            made += cur.rowcount
            if cur.rowcount: self.topics[TASK_TOPIC[t]].append(aid)
        rows = self.db.execute("SELECT id, profile FROM seeds ORDER BY id").fetchall()
        self.ids = np.array([r[0] for r in rows], np.int64)
        self.mat = np.stack([embed(r[1]) for r in rows]).astype(np.float32)
        self.pos = {int(a): i for i, a in enumerate(self.ids)}

    def retrieve(self, q, k=10, tau=0.0):
        sims = self.mat @ q.astype(np.float32)
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(self.ids[i]), float(sims[i])) for i in idx if sims[i] >= tau]

    def price(self, aid): return TIER_PRICE[TIERS[aid % len(TIERS)]]

    def instantiate(self, aid, K=5, sigma=0.05, lam=0.7) -> Agent:
        """JIT: dormant seed -> hash-sharded LLM tier + S-MMR toolset (+ MCP bridge, containerised in prod)."""
        _, _, _, _, prof, ability, price = self.db.execute("SELECT * FROM seeds WHERE id=?", (aid,)).fetchone()
        tier = TIERS[aid % len(TIERS)]
        sel = smmr(embed(prof), self.tool_vecs, K, sigma, lam, seed=aid)
        return Agent(aid, prof, tier, [self.tools[i][0] for i in sel], ability, self.llm, price=price)

# ───────────────────────── Coordination layer ─────────────────────────
@dataclass
class Node:
    id: str; desc: str; skill: str; out: str = "out.md"
    deps: list = field(default_factory=list); const: dict = field(default_factory=dict)
    status: str = "pending"; attempts: int = 0; score: float | None = None
    agent_id: object = None; cost: float = 0.0; exclude: list = field(default_factory=list)
    refine: bool = False; note: str = ""; modified: bool = False

class TaskDAG:
    def __init__(self, nodes): self.nodes = {n.id: n for n in nodes}
    def validate(self):
        """Deterministic guardrail: unknown deps + Kahn topological sort."""
        for n in self.nodes.values():
            for d in n.deps:
                if d not in self.nodes: return False, f"node {n.id} depends on unknown node {d}"
        indeg = {i: len(n.deps) for i, n in self.nodes.items()}
        succ = defaultdict(list)
        for n in self.nodes.values():
            for d in n.deps: succ[d].append(n.id)
        q, seen = deque(i for i, v in indeg.items() if v == 0), 0
        while q:
            i = q.popleft(); seen += 1
            for s in succ[i]:
                indeg[s] -= 1
                if indeg[s] == 0: q.append(s)
        if seen < len(self.nodes):
            return False, f"cyclic or unreachable nodes: {[i for i, v in indeg.items() if v > 0]}"
        return True, "ok"
    def ready(self):
        return [n for n in self.nodes.values() if n.status == "pending" and all(self.nodes[d].status == "completed" for d in n.deps)]
    def done(self): return all(n.status == "completed" for n in self.nodes.values())
    def sinks(self):
        used = {d for n in self.nodes.values() for d in n.deps}
        return [n for n in self.nodes.values() if n.id not in used]
    def replace(self, old_id, new: Node):
        """Modify primitive: executed nodes stay immutable; swap the pending node and rewire successors."""
        del self.nodes[old_id]; self.nodes[new.id] = new
        for n in self.nodes.values(): n.deps = [new.id if d == old_id else d for d in n.deps]
    def to_json(self): return json.dumps([asdict(n) for n in self.nodes.values()])
    @staticmethod
    def from_json(s): return TaskDAG([Node(**d) for d in json.loads(s)])

@dataclass
class Decision:
    action: str; strategy: str = ""; new_nodes: list = field(default_factory=list)

class Orchestrator:
    def __init__(self, llm, max_plan_tries=3): self.llm, self.max_tries = llm, max_plan_tries
    def plan(self, intent, constraints, log=print):
        """Blind planning: abstract DAG without worker identities; validate; self-correct on failure."""
        feedback = ""
        for t in range(self.max_tries):
            raw = self.llm("planner", intent, feedback=feedback, constraints=constraints)
            nodes = [Node(**{**d, "const": {**constraints, **d.get("const", {})}}) for d in json.loads(raw)["nodes"]]
            dag = TaskDAG(nodes); ok, msg = dag.validate()
            if ok: return dag
            log(f"[orchestrator] plan #{t+1} rejected by topological validation: {msg}"); feedback = msg
        raise RuntimeError("could not produce a valid DAG")
    def decide(self, dag, n, score, report, max_attempts) -> Decision:
        raw = self.llm("decide", f"Node {n.id} scored {score}. Audit: {report}\nReply JSON action=continue|retry|modify.",
                       score=score, attempts=n.attempts, max_attempts=max_attempts, node=asdict(n))
        try: d = json.loads(raw)
        except Exception: d = {"action": "retry", "strategy": "redispatch"}
        if d["action"] == "modify" and not n.modified: return Decision("modify", new_nodes=[Node(**x) for x in d["nodes"]])
        if d["action"] == "retry" and n.attempts < max_attempts: return Decision("retry", d.get("strategy", "redispatch"))
        return Decision("fallback")        # orchestrator executes the node itself

@dataclass
class Cand: aid: object; sem: float; bid: float; origin: str; const: float = 0.0

class HeuristicRanker:
    """Features: [semantic, credit, cost, constraint]. x_const dominates, as in the paper."""
    w = np.array([0.4, 0.4, -0.1, 1.0])
    def score(self, X): return X @ self.w

class LambdaMARTRanker:
    """Drop-in LTR (needs `pip install lightgbm`); train on (features, realized utility) logs."""
    def __init__(self):
        from lightgbm import LGBMRanker
        self.m = LGBMRanker(objective="lambdarank", n_estimators=100)
    def fit(self, X, y, group): self.m.fit(X, y, group=group)
    def score(self, X): return self.m.predict(X)

class Dispatcher:
    """Market dispatch: capability recruitment (vector top-k) ∪ topic bidding (pub/sub) ∪ private lanes -> LTR."""
    def __init__(self, nuwa, credit_fn, ranker=None, k=10, tau=0.0, bidders=40, seed=0):
        self.nuwa, self.credit, self.ranker = nuwa, credit_fn, ranker or HeuristicRanker()
        self.k, self.tau, self.m, self.rng = k, tau, bidders, random.Random(seed)
        self.private = defaultdict(list)     # user_id -> [RemoteAgent]
    def register_private(self, user_id, agent: RemoteAgent): self.private[user_id].append(agent)
    def _topic(self, q):
        return max(TOPIC_KW, key=lambda t: float(embed(TOPIC_KW[t]) @ q))
    def dispatch(self, node, exclude=()):
        q = embed(node.skill + " " + node.desc)
        pool = {}
        for aid, s in self.nuwa.retrieve(q, self.k, self.tau):                       # active recruitment
            pool[aid] = Cand(aid, s, self.nuwa.price(aid), "active")
        topic_ids = self.nuwa.topics[self._topic(q)]                                  # CFP on topic channel
        for aid in self.rng.sample(topic_ids, min(self.m, len(topic_ids))):
            s = float(self.nuwa.mat[self.nuwa.pos[aid]] @ q)
            if s > 0.05 and self.rng.random() < 0.6:                                  # agent opts in and bids
                bid = self.nuwa.price(aid) * self.rng.uniform(0.9, 1.2)
                pool.setdefault(aid, Cand(aid, s, bid, "bid")).bid = bid
        for ra in self.private.get(node.const.get("user_id"), []):                    # green lane
            s = float(embed(" ".join(ra.card.skills)) @ q)
            if s >= 0.3: pool[ra.id] = Cand(ra.id, s, ra.price, "private", const=1.0)
        cands = [c for c in pool.values() if c.aid not in exclude]
        if not cands: return None
        X = np.stack([[c.sem for c in cands], minmax([self.credit(c.aid) for c in cands]),
                      minmax([c.bid for c in cands]), [c.const for c in cands]], 1)
        return cands[int(np.argmax(self.ranker.score(X)))]

class Evaluator:
    """Independent blind grader; 10-point score parsed by regex with retry."""
    def __init__(self, llm, retries=3): self.llm, self.retries = llm, retries
    def audit(self, node, output):
        for _ in range(self.retries):
            txt = self.llm("evaluator", f"Task: {node.desc}\nOutput: {output}\nReply with 'SCORE: x' (0-10).")
            m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", txt)
            if m: return min(10.0, float(m.group(1))), txt
        return 0.0, "unparseable evaluator output"

class ArtifactStore:
    """Logical-path -> content; the cognitive core never sees physical storage."""
    def __init__(self): self.d = {}
    def put(self, path, content): self.d[path] = content
    def get(self, path): return self.d.get(path, "")

# ───────────────────────── Value layer ─────────────────────────
class AgentRank:
    """Signed random-walk credit: C(i)=(1-d)b + d Σ w+ C(j)/Σw+  - d Σ w- C(j)/Σw-  (d=0.85, b=100)."""
    def __init__(self, d=0.85, base=100.0, decay=0.98):
        self.d, self.b, self.decay = d, base, decay
        self.pos, self.neg, self.c = defaultdict(float), defaultdict(float), {}
    def credit(self, i):
        return self.b if i == "user" else self.c.get(i, (1 - self.d) * self.b)
    def record(self, src, dst, score, ok):
        (self.pos if ok else self.neg)[(src, dst)] += score / 10 if ok else (10 - score) / 10
    def update(self, iters=30, tol=1e-6):
        for dct in (self.pos, self.neg):
            for k in dct: dct[k] *= self.decay
        out_p, out_n, preds = defaultdict(float), defaultdict(float), defaultdict(set)
        for (j, i), w in self.pos.items(): out_p[j] += w; preds[i].add(j)
        for (j, i), w in self.neg.items(): out_n[j] += w; preds[i].add(j)
        for _ in range(iters):
            new, delta = {}, 0.0
            for i, js in preds.items():
                v = (1 - self.d) * self.b
                for j in js:
                    cj = self.credit(j)
                    if out_p[j] > 0: v += self.d * self.pos.get((j, i), 0) * cj / out_p[j]
                    if out_n[j] > 0: v -= self.d * self.neg.get((j, i), 0) * cj / out_n[j]
                new[i] = v; delta = max(delta, abs(v - self.credit(i)))
            self.c.update(new)
            if delta < tol: break

class Ledger:
    """Entry -> task creation (freeze) -> settlement (4 parties) -> locked/withdrawable capital pool."""
    def __init__(self, orch_fee=0.05, platform_fee=0.03, lock_ratio=0.5):
        self.acc = defaultdict(lambda: {"free": 0.0, "locked": 0.0, "frozen": 0.0})
        self.of, self.pf, self.lock = orch_fee, platform_fee, lock_ratio; self.escrow = {}
    def deposit(self, who, amt): self.acc[who]["free"] += amt
    def freeze(self, user, amt, task):
        assert self.acc[user]["free"] >= amt, "insufficient funds"
        self.acc[user]["free"] -= amt; self.escrow[task] = (user, amt)
    def settle(self, task, orch, payments):
        user, budget = self.escrow.pop(task)
        cost = sum(payments.values()); fo, fp = cost * self.of, cost * self.pf; total = cost + fo + fp
        self.acc[user]["free"] += max(0.0, budget - total)
        if total > budget: self.acc[orch]["free"] -= total - budget          # orchestrator liable for overrun
        for a, p in payments.items():
            u = min(self.acc[a]["locked"], p)                                 # participation unlocks earlier earnings
            self.acc[a]["locked"] -= u; self.acc[a]["free"] += u + p * (1 - self.lock); self.acc[a]["locked"] += p * self.lock
        self.acc[orch]["free"] += fo; self.acc["platform"]["free"] += fp
        return dict(cost=cost, orchestrator_fee=fo, platform_fee=fp, refund=max(0.0, budget - total))

class MandateChain:
    """Lightweight hash chain of signed phase mandates (HMAC stands in for real signatures)."""
    @staticmethod
    def _key(e): return hashlib.sha256(b"holos-demo-key:" + e.encode()).digest()
    def __init__(self): self.blocks = []
    def append(self, phase, payload, signers):
        prev = self.blocks[-1]["hash"] if self.blocks else "0" * 64
        body = json.dumps({"phase": phase, "payload": payload, "prev": prev}, sort_keys=True, default=str)
        sigs = {s: hmac.new(self._key(s), body.encode(), "sha256").hexdigest() for s in signers}
        h = hashlib.sha256((body + json.dumps(sigs, sort_keys=True)).encode()).hexdigest()
        self.blocks.append({"body": body, "sigs": sigs, "hash": h})
    def verify(self):
        prev = "0" * 64
        for b in self.blocks:
            if json.loads(b["body"])["prev"] != prev: return False
            if any(not hmac.compare_digest(s, hmac.new(self._key(e), b["body"].encode(), "sha256").hexdigest()) for e, s in b["sigs"].items()): return False
            if hashlib.sha256((b["body"] + json.dumps(b["sigs"], sort_keys=True)).encode()).hexdigest() != b["hash"]: return False
            prev = b["hash"]
        return True

# ───────────────────────── Holos engine ─────────────────────────
class Holos:
    def __init__(self, llm=None, n_agents=2000, seed=0, threshold=6.0, max_attempts=4):
        self.llm = llm or MockLLM(seed)
        self.nuwa = Nuwa(self.llm, seed=seed); self.nuwa.synthesize(n_agents)
        self.rank, self.ledger, self.chain = AgentRank(), Ledger(), MandateChain()
        self.orch, self.evaluator = Orchestrator(self.llm), Evaluator(self.llm)
        self.dispatcher = Dispatcher(self.nuwa, self.rank.credit, seed=seed)
        self.threshold, self.max_attempts = threshold, max_attempts
        self.wf = sqlite3.connect(":memory:", check_same_thread=False)
        self.wf.execute("CREATE TABLE wf(sid TEXT PRIMARY KEY, state TEXT)")
        self.events, self._tasks = [], set()
    def log(self, msg): self.events.append(msg); print(msg)
    def _save(self, sid, dag): self.wf.execute("REPLACE INTO wf VALUES(?,?)", (sid, dag.to_json())); self.wf.commit()
    def _load(self, sid): return TaskDAG.from_json(self.wf.execute("SELECT state FROM wf WHERE sid=?", (sid,)).fetchone()[0])
    def _agent(self, aid):
        for lst in self.dispatcher.private.values():
            for ra in lst:
                if ra.id == aid: return ra
        return self.nuwa.instantiate(aid)
    def _self_agent(self): return Agent("orchestrator", "Orchestrator fallback", "frontier", [], 0.7, self.llm)

    def _launch(self, dag, n, q, sid, store):
        if n.refine and n.agent_id is not None: aid = n.agent_id                    # refinement: same agent
        else:
            c = self.dispatcher.dispatch(n, exclude=set(n.exclude))
            if c is None: agent, n.cost = self._self_agent(), 0.0; aid = "orchestrator"
            else: aid, n.cost = c.aid, c.bid; self.log(f"[dispatch] {n.id} -> agent {aid} ({c.origin}, sem={c.sem:.2f}, bid={c.bid:.2f})")
        agent = self._self_agent() if aid == "orchestrator" else self._agent(aid)
        if isinstance(agent, RemoteAgent):     # per-launch copy carrying identity/lineage for the A2A proxy
            agent = copy.copy(agent)
            agent.ctx = {"sid": sid, "node": n.id, "lineage": [str(dag.nodes[d].agent_id) for d in n.deps]}
        n.agent_id, n.status = aid, "running"
        inputs = [store.get(f"/{sid}/{d}/{dag.nodes[d].out}") for d in n.deps]
        directive = self.llm("directive", n.desc, profile=agent.profile, io=n.out) + f"\nInputs: {inputs}"
        if n.refine: directive += f"\nRefine according to audit: {n.note}"
        async def work():
            out = await asyncio.to_thread(agent.run, directive)
            await q.put((n.id, out))
        t = asyncio.create_task(work()); self._tasks.add(t); t.add_done_callback(self._tasks.discard)

    async def run(self, intent, budget=40.0, user="alice", constraints=None):
        sid = f"s{int(time.time() * 1000) % 10**7}"; orch = "orchestrator:platform"
        self.ledger.deposit(user, budget * 2); self.ledger.freeze(user, budget, sid)           # entry + task creation
        self.chain.append("task_created", {"sid": sid, "budget": budget}, [user, "platform"])
        dag = self.orch.plan(intent, constraints or {}, self.log)
        self.chain.append("orchestrated", {"nodes": list(dag.nodes)}, [orch])
        store, q, payments = ArtifactStore(), asyncio.Queue(), defaultdict(float)
        self._save(sid, dag)
        while not dag.done():
            running = [n for n in dag.nodes.values() if n.status == "running"]
            for n in dag.ready(): self._launch(dag, n, q, sid, store); running.append(n)
            self._save(sid, dag)                                   # serialize state, then go dormant
            if not running: raise RuntimeError("deadlock")
            nid, out = await q.get()                               # completion event wakes the loop
            dag = self._load(sid); n = dag.nodes[nid]              # rehydrate session
            n.attempts += 1; n.refine = False
            score, report = self.evaluator.audit(n, out); n.score = score
            ok = score >= self.threshold
            for src in (dag.nodes[d].agent_id for d in n.deps) or ["user"]: self.rank.record(src, n.agent_id, score, ok)
            self.log(f"[eval] {n.id} by {n.agent_id}: {score:.1f}/10 (attempt {n.attempts}) {'OK' if ok else 'FAIL'}")
            if ok:
                store.put(f"/{sid}/{n.id}/{n.out}", out); n.status = "completed"; payments[n.agent_id] += n.cost
                self.chain.append("executed", {"node": n.id, "agent": n.agent_id, "score": score}, [str(n.agent_id), orch])
            else:
                d = self.orch.decide(dag, n, score, report, self.max_attempts)
                if d.action == "retry":
                    n.status, n.refine, n.note = "pending", d.strategy == "refine", report
                    if d.strategy == "redispatch": n.exclude.append(n.agent_id)
                    self.log(f"[orchestrator] retry({d.strategy}) {n.id}")
                elif d.action == "modify":
                    self.log(f"[orchestrator] modify: {n.id} -> {d.new_nodes[0].id}"); dag.replace(n.id, d.new_nodes[0])
                else:
                    self.log(f"[orchestrator] fallback: executing {n.id} itself")
                    out = self._self_agent().run(n.desc); store.put(f"/{sid}/{n.id}/{n.out}", out)
                    n.status, n.agent_id, n.cost = "completed", "orchestrator", 0.0
            self._save(sid, dag)
        settle = self.ledger.settle(sid, orch, {a: p for a, p in payments.items() if a != "orchestrator"})
        self.chain.append("settled", settle, ["platform", orch])
        self.rank.update()                                          # periodic AgentRank window
        result = [store.get(f"/{sid}/{n.id}/{n.out}") for n in dag.sinks()]
        return dict(session=sid, result=result, settlement=settle, chain_valid=self.chain.verify(),
                    nodes={n.id: (n.agent_id, n.score, n.attempts) for n in dag.nodes.values()})

# ───────────────────────── Economic simulation (paper §4.3.1, scaled down) ─────────────────────────
def simulate_market(n=30, epochs=200, tasks=100, temp=0.15, seed=0, malicious=0):
    rng = np.random.default_rng(seed)
    ability, price, revenue = rng.uniform(0.2, 0.95, n), rng.uniform(1, 3, n), np.zeros(n)
    rank, ranker = AgentRank(), HeuristicRanker()
    for _ in range(epochs):
        cred = minmax([rank.credit(i) for i in range(n)]); cost = minmax(price)
        X = np.stack([np.ones(n), cred, cost, np.zeros(n)], 1)
        logits = ranker.score(X) / temp; p = np.exp(logits - logits.max()); p /= p.sum()
        for i in rng.choice(n, size=tasks, p=p):
            s = 10 * float(np.clip(ability[i] + rng.normal(0, 0.1), 0, 1)); ok = s >= 6
            rank.record("user", int(i), s, ok); revenue[i] += price[i] if ok else 0
        rank.update()
    return dict(spearman_revenue_vs_ability=spearman(revenue, ability),
                spearman_credit_vs_ability=spearman([rank.credit(i) for i in range(n)], ability))

# ───────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "sim":
        print(simulate_market())
    else:
        h = Holos(n_agents=2000)
        h.dispatcher.register_private("alice", RemoteAgent(
            AgentCard("alice-writer", "https://alice.example/a2a", ["technical report writing"]),
            lambda directive: "FINAL: private writer output [quality=0.93]", price=2.0))
        res = asyncio.run(h.run("edge-AI chip market analysis", budget=40, constraints={"user_id": "alice"}))
        print(json.dumps({k: v for k, v in res.items() if k != "result"}, indent=2, default=str))
        print("ledger:", {k: {a: round(b, 2) for a, b in v.items()} for k, v in h.ledger.acc.items() if k != "user" or True})
