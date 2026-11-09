"""FastAPI + WebSocket game server. Localhost only (127.0.0.1).

    python -m sim.server                  # http://127.0.0.1:8765  (the Vite client proxies /ws and /api here)

One session per WebSocket connection. The session runs in its own thread at real time (catching up after a slow
tick, e.g. an autonomy CBS replan); snapshots are queued to the socket. Client -> server messages:
  {"type": "start", "mode": "play"|"watch"|"demo", "map": "p1" | "demo-lakes-1", "seed": 1, "seekers": 1}
  {"type": "input", "x", "y", "z", "yaw", "v", "drop"}     player pose (client-side flight model)
  {"type": "plan", "plan": {drone_id: {"x", "y"} | {"hotspot": id}}}
  {"type": "skip"} | {"type": "pause"} | {"type": "resume"} | {"type": "restart", "seed"?} | {"type": "stop"}
  {"type": "follow", "id": drone_id | null}                  DEMO / watch: whose thermal feed to stream
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from dataclasses import replace

from sim.config import DEMO_MAPS, GameConfig, demo_config
from sim.game import Session
from sim.maps import available_maps

HOST, PORT = "127.0.0.1", int(os.environ.get("SIM_PORT", "8765"))   # localhost only (global rule)

app = FastAPI(title="Pandas wildfire swarm sim")


@app.get("/api/maps")
def maps() -> dict:
    base = available_maps()
    ok = {m["id"]: m["available"] for m in base}
    demo = [{"id": k, "title": v["title"], "available": True, "wet": v["wet"], "real": ok.get(v["tiles"], False)}
            for k, v in DEMO_MAPS.items()]
    return {"maps": base, "demo_maps": demo}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


class Runner(threading.Thread):
    """Steps one Session at real time and pushes snapshots to an asyncio queue."""

    def __init__(self, params: dict, loop: asyncio.AbstractEventLoop, out: asyncio.Queue):
        super().__init__(daemon=True)
        self.params, self.loop, self.out = params, loop, out
        self.lock = threading.Lock()
        self.stop_flag = False
        self.session: Session | None = None
        self.pending: list[tuple[str, dict]] = []
        self.next_t = 0.0

    def push(self, msg: dict) -> None:
        # allow_nan=False: browsers' JSON.parse rejects NaN/Infinity, so fail loudly here instead of freezing the UI
        self.loop.call_soon_threadsafe(self.out.put_nowait, json.dumps(msg, separators=(",", ":"), allow_nan=False))

    def command(self, kind: str, msg: dict) -> None:
        with self.lock:
            if kind == "input":   # only the newest pose matters
                self.pending = [p for p in self.pending if p[0] != "input"]
            self.pending.append((kind, msg))

    def _build(self) -> None:
        p = self.params
        self.push({"type": "loading", "map": p.get("map", "p1")})
        if p.get("mode") == "demo":
            info = DEMO_MAPS.get(p.get("map", ""), DEMO_MAPS["demo-lakes-1"])
            cfg = demo_config(wet=info["wet"])
            if p.get("fast"):
                cfg = replace(cfg, demo_s=40.0)
        else:
            cfg = GameConfig()
            if p.get("fast"):   # dev/testing: short phases (same physics)
                cfg = replace(cfg, briefing_s=2.0, seek_s=20.0, plan_s=12.0, extinguish_s=25.0)
        self.session = Session(map_id=p.get("map", "p1"), seed=int(p.get("seed", 1)), mode=p.get("mode", "play"),
                               swarm_seekers=int(p.get("seekers", 1)), cfg=cfg)
        self.push(self.session.init_message())

    def run(self) -> None:
        try:
            self._build()
            dt = 1.0 / self.session.cfg.tick_hz
            self.next_t = time.perf_counter()
            while not self.stop_flag:
                with self.lock:
                    cmds, self.pending = self.pending, []
                for kind, msg in cmds:
                    self._apply(kind, msg)
                s = self.session
                if s.paused or s.phase == "SCORE":
                    self.push(s.snapshot())
                    time.sleep(0.1)
                    self.next_t = time.perf_counter()
                    continue
                now = time.perf_counter()
                if now < self.next_t:
                    time.sleep(min(self.next_t - now, 0.05))
                    continue
                s.step()
                self.next_t += dt
                if time.perf_counter() - self.next_t > 1.0:     # far behind (long autonomy replan): don't burst forever
                    self.next_t = time.perf_counter() - 0.25
                self.push(s.snapshot())
        except Exception as exc:  # surface sim errors in the client instead of silently dying
            traceback.print_exc()
            self.push({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def _apply(self, kind: str, msg: dict) -> None:
        s = self.session
        if kind == "input":
            s.player_input(msg)
        elif kind == "plan":
            s.set_plan(msg.get("plan", {}))
        elif kind == "skip":
            s.skip()
        elif kind == "pause":
            s.paused = True
        elif kind == "resume":
            s.paused = False
        elif kind == "follow":
            s.follow_id = msg.get("id")
        elif kind == "restart":
            if "seed" in msg:
                self.params["seed"] = int(msg["seed"])
            self._build()
            self.next_t = time.perf_counter()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue()
    runner: Runner | None = None

    async def sender() -> None:
        while True:
            text = await out.get()
            await ws.send_text(text)

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "start":
                if runner is not None:
                    runner.stop_flag = True
                while not out.empty():
                    out.get_nowait()
                runner = Runner(dict(msg), loop, out)
                runner.start()
            elif kind == "stop":
                if runner is not None:
                    runner.stop_flag = True
                    runner = None
            elif runner is not None:
                runner.command(kind, msg)
    except WebSocketDisconnect:
        pass
    finally:
        if runner is not None:
            runner.stop_flag = True
        send_task.cancel()


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", ws_max_size=4 * 1024 * 1024)


if __name__ == "__main__":
    main()
