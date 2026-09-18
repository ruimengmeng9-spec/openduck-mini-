"""Small browser UI for live Open Duck Mini simulation and MiniCPM-o control."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mujoco
import numpy as np
from PIL import Image

from open_duck_agent.agent import run_minicpmo_turn
from open_duck_agent.duck_sim import DuckSimulation
from open_duck_agent.tools import RobotTools


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open Duck 实时控制台</title>
<style>
:root{color-scheme:dark;--bg:#101318;--panel:#191e26;--line:#2b3340;--gold:#ffc928;--ok:#45d483;--bad:#ff6673}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef2f7;font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:auto;padding:22px}.top{display:flex;align-items:center;gap:12px;margin-bottom:16px}
h1{font-size:22px;margin:0}.dot{width:11px;height:11px;border-radius:50%;background:#77808d}.dot.ok{background:var(--ok);box-shadow:0 0 12px var(--ok)}
.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:16px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
.view{padding:0;overflow:hidden;min-height:420px;display:grid;place-items:center;background:#050607}.view img{display:block;width:100%;height:auto}
.row{display:flex;gap:9px}.row input{flex:1;background:#0d1015;color:#fff;border:1px solid #3b4656;border-radius:9px;padding:12px;font-size:15px}
button{border:0;border-radius:9px;background:var(--gold);color:#151515;font-weight:700;padding:0 18px;cursor:pointer}button:disabled{opacity:.5;cursor:not-allowed}
.examples{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.examples button{background:#29313d;color:#e8edf4;padding:7px 10px;font-weight:500}
.metric{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding:8px 0}.metric span:last-child{font-family:ui-monospace,monospace;text-align:right}
#message{min-height:48px;margin:12px 0;color:#cbd3de}.good{color:var(--ok)!important}.bad{color:var(--bad)!important}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;color:#aeb8c6;max-height:220px;overflow:auto}
@media(max-width:800px){.grid{grid-template-columns:1fr}.view{min-height:240px}}
</style></head>
<body><main>
<div class="top"><span id="dot" class="dot"></span><h1>Open Duck 实时控制台</h1><span id="mode">正在连接…</span></div>
<div class="grid">
  <section class="panel view"><img src="/stream.mjpg" alt="Open Duck live view"></section>
  <aside class="panel">
    <div class="row"><input id="prompt" value="先站立1秒，以0.1米每秒向前走2秒，向左转1秒，最后停止"><button id="send">执行</button></div>
    <div class="examples"><button data-p="站立2秒，然后停止">站立</button><button data-p="以0.1米每秒向前走3秒，然后停止">前进</button><button data-p="向左转2秒，然后停止">左转</button><button data-p="立即停止">停止</button></div>
    <div id="message">等待指令。</div>
    <div class="metric"><span>仿真时间</span><span id="simtime">--</span></div>
    <div class="metric"><span>当前动作</span><span id="action">--</span></div>
    <div class="metric"><span>位置 (m)</span><span id="position">--</span></div>
    <div class="metric"><span>局部速度 (m/s)</span><span id="velocity">--</span></div>
    <div class="metric"><span>竖直度</span><span id="upz">--</span></div>
    <div class="metric"><span>安全状态</span><span id="fallen">--</span></div>
    <pre id="detail"></pre>
  </aside>
</div></main>
<script>
const $=id=>document.getElementById(id);let currentJob=null;
document.querySelectorAll('[data-p]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.p});
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter')submit()});$('send').onclick=submit;
async function submit(){const prompt=$('prompt').value.trim();if(!prompt)return;$('send').disabled=true;$('message').className='';$('message').textContent='指令已提交，MiniCPM-o 正在规划…';
 try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt})});const j=await r.json();if(!r.ok)throw Error(j.error||r.statusText);currentJob=j.job_id}
 catch(e){$('message').className='bad';$('message').textContent='提交失败：'+e.message;$('send').disabled=false}}
function vec(v){return Array.isArray(v)?v.map(x=>Number(x).toFixed(3)).join(', '):'--'}
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store'}),j=await r.json();$('dot').className='dot ok';$('mode').textContent=j.busy?'正在执行':'已连接';
 const s=j.status||{};$('simtime').textContent=s.simulation_time_s==null?'--':Number(s.simulation_time_s).toFixed(2)+' s';$('action').textContent=s.last_action||'--';$('position').textContent=vec(s.base_position_m);$('velocity').textContent=vec(s.local_velocity_mps);$('upz').textContent=s.up_vector_z==null?'--':Number(s.up_vector_z).toFixed(4);$('fallen').textContent=s.fallen?'已摔倒':'正常';$('fallen').className=s.fallen?'bad':'good';
 if(currentJob){const x=await fetch('/api/job/'+currentJob,{cache:'no-store'}),job=await x.json();if(job.state==='done'||job.state==='error'){$('send').disabled=false;$('message').className=job.state==='done'?'good':'bad';$('message').textContent=job.state==='done'?'执行完成，机器人已停止。':'执行失败：'+job.error;$('detail').textContent=JSON.stringify(job.report||job,null,2);currentJob=null}else{$('message').textContent=job.state==='planning'?'MiniCPM-o 正在规划…':'动作正在实时执行…'}}
 }catch(e){$('dot').className='dot';$('mode').textContent='连接中断'}setTimeout(poll,300)}poll();
</script></body></html>"""


@dataclass
class Job:
    prompt: str
    state: str = "queued"
    report: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class LiveDuckSimulation(DuckSimulation):
    """DuckSimulation variant that publishes frames and respects wall time."""

    def __init__(self, *args: Any, runtime: "Runtime", **kwargs: Any) -> None:
        self.runtime = runtime
        super().__init__(*args, warmup_s=0.0, **kwargs)
        # The repository XML uses MuJoCo's default 640 px offscreen framebuffer.
        self.renderer = mujoco.Renderer(self.sim.model, height=360, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 0.85
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0
        self._publish_frame(force=True)
        self.stand(3.0, source="startup_warmup")

    def _publish_frame(self, force: bool = False) -> None:
        if not force and self._physics_steps % (self.sim.decimation * 2) != 0:
            return
        base = np.asarray(self.sim.get_floating_base_qpos(self.sim.data.qpos), dtype=float)
        self.camera.lookat[:] = base[:3]
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        rgb = self.renderer.render()
        out = io.BytesIO()
        Image.fromarray(rgb).save(out, format="JPEG", quality=82)
        self.runtime.publish(out.getvalue(), self.status())

    def _run_command(
        self,
        action_name: str,
        command: list[float],
        duration_s: float,
        source: str = "tool",
    ) -> dict[str, Any]:
        duration = self._clip_finite(duration_s, self.CONTROL_DT, self.MAX_DURATION_S, "duration_s")
        control_steps = max(1, int(round(duration / self.CONTROL_DT)))
        sim = self.sim
        sim.commands = [float(x) for x in command]
        self._last_command = np.asarray(command, dtype=np.float32)
        self._last_action_name = action_name
        start_time = float(sim.data.time)
        start_position = np.asarray(sim.get_floating_base_qpos(sim.data.qpos)[:3], dtype=float).copy()
        wall_start = time.monotonic()
        interrupted = False

        for control_index in range(control_steps):
            for _ in range(sim.decimation):
                mujoco.mj_step(sim.model, sim.data)
                self._physics_steps += 1

            sim.imitation_i = (sim.imitation_i + sim.phase_frequency_factor) % sim.PRM.nb_steps_in_period
            angle = sim.imitation_i / sim.PRM.nb_steps_in_period * 2.0 * np.pi
            sim.imitation_phase = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            observation = sim.get_obs(sim.data, sim.commands)
            action = np.asarray(sim.policy.infer(observation), dtype=np.float32)
            sim.last_last_last_action = sim.last_last_action.copy()
            sim.last_last_action = sim.last_action.copy()
            sim.last_action = action.copy()
            target = sim.default_actuator + action * sim.action_scale
            max_change = sim.max_motor_velocity * (sim.sim_dt * sim.decimation)
            sim.motor_targets = np.clip(target, sim.prev_motor_targets - max_change, sim.prev_motor_targets + max_change)
            sim.prev_motor_targets = sim.motor_targets.copy()
            sim.data.ctrl[:] = sim.motor_targets
            self._publish_frame()

            state = self.status()
            if state["fallen"]:
                interrupted = True
                sim.commands = [0.0] * 7
                self._last_command[:] = 0.0
                break
            remaining = wall_start + (control_index + 1) * self.CONTROL_DT - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        self._publish_frame(force=True)
        end = self.status()
        self.runtime.publish_status(end)
        end_position = np.asarray(end["base_position_m"], dtype=float)
        result = {
            "ok": not interrupted,
            "action": action_name,
            "requested_duration_s": round(duration, 3),
            "executed_duration_s": round(float(sim.data.time) - start_time, 6),
            "command": [round(float(x), 6) for x in command],
            "horizontal_displacement_m": round(float(np.linalg.norm(end_position[:2] - start_position[:2])), 6),
            **end,
        }
        if interrupted:
            result["error"] = "Safety stop: the robot is fallen or state is not finite."
        self._log({"event": "action", "source": source, **result})
        return result


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.frames = threading.Condition()
        self.latest_frame: bytes | None = None
        self.latest_status: dict[str, Any] = {}
        self.frame_number = 0
        self.jobs: dict[str, Job] = {}
        self.jobs_lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue(maxsize=8)
        self.ready = threading.Event()
        self.startup_error: str | None = None
        self.busy = False
        threading.Thread(target=self._worker, name="duck-runtime", daemon=True).start()

    def publish(self, frame: bytes, status: dict[str, Any]) -> None:
        with self.frames:
            self.latest_frame = frame
            self.latest_status = copy.deepcopy(status)
            self.frame_number += 1
            self.frames.notify_all()

    def publish_status(self, status: dict[str, Any]) -> None:
        with self.frames:
            self.latest_status = copy.deepcopy(status)

    def submit(self, prompt: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.jobs_lock:
            self.jobs[job_id] = Job(prompt=prompt)
        try:
            self.queue.put_nowait(job_id)
        except queue.Full:
            with self.jobs_lock:
                self.jobs.pop(job_id, None)
            raise RuntimeError("指令队列已满，请稍后重试")
        return job_id

    def job_payload(self, job_id: str) -> dict[str, Any] | None:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return {"job_id": job_id, "state": job.state, "report": job.report, "error": job.error}

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self.jobs_lock:
            job = self.jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)

    def _worker(self) -> None:
        try:
            controller = LiveDuckSimulation(
                repo_root=self.args.repo_root,
                onnx_model=self.args.onnx_model,
                output_root=self.args.output_root,
                runtime=self,
            )
            tools = RobotTools(controller)
            self.ready.set()
            while True:
                job_id = self.queue.get()
                job = self.jobs[job_id]
                self.busy = True
                self._set_job(job_id, state="planning")
                try:
                    report = run_minicpmo_turn(
                        tools,
                        job.prompt,
                        repo_root=self.args.repo_root,
                        root=self.args.root,
                        planner_python=self.args.minicpmo_python,
                        model_path=self.args.minicpmo_model,
                        gpu=self.args.minicpmo_gpu,
                        timeout_s=self.args.planner_timeout,
                    )
                    self._set_job(job_id, state="done", report=report)
                except Exception as exc:
                    try:
                        controller.stop(settle_s=0.5, source="live_server_error")
                    except Exception:
                        pass
                    self._set_job(job_id, state="error", error=f"{type(exc).__name__}: {exc}")
                finally:
                    self.busy = False
                    self.queue.task_done()
        except Exception as exc:
            self.startup_error = f"{type(exc).__name__}: {exc}"
            self.ready.set()


class Handler(BaseHTTPRequestHandler):
    runtime: Runtime

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            self._json({"ready": self.runtime.ready.is_set() and not self.runtime.startup_error, "busy": self.runtime.busy, "startup_error": self.runtime.startup_error, "status": self.runtime.latest_status})
            return
        if path.startswith("/api/job/"):
            job = self.runtime.job_payload(path.rsplit("/", 1)[-1])
            self._json(job or {"error": "job not found"}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
            return
        if path == "/stream.mjpg":
            self._stream()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/command":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError("请求大小无效")
            payload = json.loads(self.rfile.read(length))
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt or len(prompt) > 500:
                raise ValueError("指令必须为 1 到 500 个字符")
            if self.runtime.startup_error:
                raise RuntimeError(self.runtime.startup_error)
            job_id = self.runtime.submit(prompt)
            self._json({"job_id": job_id}, HTTPStatus.ACCEPTED)
        except (ValueError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = -1
        try:
            while True:
                with self.runtime.frames:
                    self.runtime.frames.wait_for(lambda: self.runtime.frame_number != seen, timeout=3.0)
                    frame = self.runtime.latest_frame
                    seen = self.runtime.frame_number
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def build_parser() -> argparse.ArgumentParser:
    root = Path(os.environ.get("OPEN_DUCK_ROOT", "/data/shijinsheng/open_duck"))
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--repo-root", type=Path, default=root / "projects/Open_Duck_Playground")
    p.add_argument("--onnx-model", type=Path, default=root / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx")
    p.add_argument("--output-root", type=Path, default=root / "outputs")
    p.add_argument("--minicpmo-python", type=Path, default=root / "minicpmo/.venv/bin/python")
    p.add_argument("--minicpmo-model", type=Path, default=root / "models/MiniCPM-o-4_5-awq")
    p.add_argument("--minicpmo-gpu", default=os.environ.get("MINICPMO_GPU", "7"))
    p.add_argument("--planner-timeout", type=float, default=900.0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    runtime = Runtime(args)
    Handler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open Duck live UI: http://{args.host}:{args.port}", flush=True)
    print("等待仿真初始化；请通过 VS Code 转发该端口。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止实时控制台。", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
