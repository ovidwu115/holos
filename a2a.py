#!/usr/bin/env python3
"""
Minimal A2A transport for Holos (paper §3.3: Agent Card discovery + proxy with identity/lineage).

Implemented subset (JSON-RPC 2.0 over HTTP, stdlib only):
  GET  /.well-known/agent-card.json      (also /.well-known/agent.json for older peers)
  POST /a2a   message/send  (blocking or non-blocking), tasks/get
Holos additions carried in message.metadata["holos"]: {sid, node, lineage, sig}
  - sig = HMAC-SHA256 identity assertion by the platform proxy; servers may verify it.
Not implemented: message/stream (SSE), tasks/cancel, push notifications, OAuth/mTLS, file parts,
artifact transfer handles.  Field names follow the public A2A spec as far as I know it; check
against the spec version your peers speak before interop use.
"""
from __future__ import annotations
import asyncio, hashlib, hmac, json, threading, time, urllib.request, urllib.error, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import holos

PROTO = "0.3.0"

def _sig(key: bytes, sid, node, lineage):
    msg = json.dumps([sid, node, lineage], sort_keys=True).encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

class A2AError(Exception):
    def __init__(self, code, message): super().__init__(f"[{code}] {message}"); self.code = code

# ─────────────── server ───────────────
class A2AServer:
    def __init__(self, name, skills, handler, port=0, price=1.0, token=None, verify_key=None, host="127.0.0.1"):
        """handler(text, meta) -> str.  skills: list of (id, description, tags)."""
        self.name, self.skills, self.handler, self.price = name, skills, handler, price
        self.token, self.verify_key = token, verify_key
        self.tasks, self.received, outer = {}, [], self
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            def do_GET(self):
                if self.path in ("/.well-known/agent-card.json", "/.well-known/agent.json"): self._send(200, outer.card())
                else: self._send(404, {"error": "not found"})
            def do_POST(self):
                if outer.token and self.headers.get("Authorization") != f"Bearer {outer.token}":
                    return self._send(401, {"error": "unauthorized"})
                try: req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                except Exception: return self._send(200, outer._err(None, -32700, "parse error"))
                self._send(200, outer.rpc(req))
        self.httpd = ThreadingHTTPServer((host, port), H)
        self.base = f"http://{host}:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def card(self):
        url = self.base + "/a2a"
        return {"protocolVersion": PROTO, "name": self.name, "description": f"{self.name} (Holos demo agent)",
                "url": url, "preferredTransport": "JSONRPC", "version": "0.1.0",
                "supportedInterfaces": [{"url": url, "protocolBinding": "JSONRPC"}],
                "capabilities": {"streaming": False}, "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"],
                "skills": [{"id": i, "name": i, "description": d, "tags": t} for i, d, t in self.skills],
                "x-holos": {"price": self.price}}

    @staticmethod
    def _err(i, code, msg): return {"jsonrpc": "2.0", "id": i, "error": {"code": code, "message": msg}}
    @staticmethod
    def _ok(i, res): return {"jsonrpc": "2.0", "id": i, "result": res}

    def _task(self, tid, state, text=None, err=None):
        t = {"kind": "task", "id": tid, "contextId": tid, "status": {"state": state, "timestamp": time.time()}}
        if text is not None: t["artifacts"] = [{"artifactId": uuid.uuid4().hex, "name": "result", "parts": [{"kind": "text", "text": text}]}]
        if err: t["status"]["message"] = {"role": "agent", "parts": [{"kind": "text", "text": err}]}
        return t

    def _run(self, tid, text, meta):
        try: self.tasks[tid] = self._task(tid, "completed", self.handler(text, meta))
        except Exception as e: self.tasks[tid] = self._task(tid, "failed", err=str(e))

    def rpc(self, req):
        i, m, p = req.get("id"), req.get("method"), req.get("params") or {}
        if m == "message/send":
            msg = p.get("message") or {}
            text = " ".join(x.get("text", "") for x in msg.get("parts", []) if x.get("kind") == "text")
            meta = (msg.get("metadata") or {}).get("holos")
            if self.verify_key is not None:                      # identity assertion check
                if not meta or not hmac.compare_digest(meta.get("sig", ""), _sig(self.verify_key, meta.get("sid"), meta.get("node"), meta.get("lineage"))):
                    return self._err(i, -32602, "invalid or missing identity assertion")
            self.received.append(meta)
            tid = uuid.uuid4().hex
            if (p.get("configuration") or {}).get("blocking", True):
                self._run(tid, text, meta)
            else:
                self.tasks[tid] = self._task(tid, "working")
                threading.Thread(target=self._run, args=(tid, text, meta), daemon=True).start()
            return self._ok(i, self.tasks[tid])
        if m == "tasks/get":
            t = self.tasks.get(p.get("id"))
            return self._ok(i, t) if t else self._err(i, -32001, "task not found")
        return self._err(i, -32601, f"method not found: {m}")

# ─────────────── client ───────────────
class A2AClient:
    def __init__(self, token=None, signing_key=None, timeout=10):
        self.token, self.key, self.timeout = token, signing_key, timeout
    def _req(self, url, data=None):
        h = {"Content-Type": "application/json"}
        if self.token: h["Authorization"] = f"Bearer {self.token}"
        r = urllib.request.Request(url, json.dumps(data).encode() if data is not None else None, h)
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as f: return json.loads(f.read())
        except urllib.error.HTTPError as e: raise A2AError(e.code, e.reason)
    def discover(self, base):
        for path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
            try: return self._req(base.rstrip("/") + path)
            except Exception: continue
        raise A2AError(-1, f"no agent card at {base}")
    def rpc(self, url, method, params):
        r = self._req(url, {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params})
        if "error" in r: raise A2AError(r["error"]["code"], r["error"]["message"])
        return r["result"]
    def send(self, url, text, ctx=None, blocking=False, poll=0.05, max_wait=30):
        msg = {"role": "user", "messageId": uuid.uuid4().hex, "parts": [{"kind": "text", "text": text}]}
        if ctx and self.key:
            msg["metadata"] = {"holos": {**ctx, "sig": _sig(self.key, ctx["sid"], ctx["node"], ctx["lineage"])}}
        t = self.rpc(url, "message/send", {"message": msg, "configuration": {"blocking": blocking}})
        t0 = time.time()
        while t["status"]["state"] in ("submitted", "working"):   # long-horizon: poll tasks/get
            if time.time() - t0 > max_wait: raise A2AError(-2, "task timeout")
            time.sleep(poll); t = self.rpc(url, "tasks/get", {"id": t["id"]})
        if t["status"]["state"] != "completed":
            raise A2AError(-3, f"task {t['status']['state']}: {t['status'].get('message')}")
        return " ".join(p["text"] for a in t.get("artifacts", []) for p in a["parts"] if p.get("kind") == "text")

# ─────────────── Holos integration ───────────────
class A2ARemoteAgent(holos.RemoteAgent):
    """User-owned remote agent reached over A2A; Holos' proxy attaches identity + lineage."""
    def __init__(self, client: A2AClient, card: dict):
        skills = [f"{s['name']} {s.get('description','')} {' '.join(s.get('tags', []))}" for s in card.get("skills", [])]
        super().__init__(holos.AgentCard(card["name"], card["url"], skills), None, price=card.get("x-holos", {}).get("price", 1.0))
        self.client, self.url = client, card["url"]
    def run(self, directive, max_steps=3):
        return self.client.send(self.url, directive, ctx=self.ctx or None)

async def probe_and_register(h: holos.Holos, user_id, base_urls, client: A2AClient):
    """Asynchronous probing of Agent Cards -> private 'green lane' for the user."""
    cards = await asyncio.gather(*(asyncio.to_thread(client.discover, u) for u in base_urls), return_exceptions=True)
    ok = []
    for u, c in zip(base_urls, cards):
        if isinstance(c, Exception): h.log(f"[a2a] probe failed {u}: {c}"); continue
        h.dispatcher.register_private(user_id, A2ARemoteAgent(client, c)); ok.append(c["name"])
    h.log(f"[a2a] registered for {user_id}: {ok}"); return ok
