# Open Duck Mini 大模型动作接口

这个模块在现有 Open Duck Mini 原生 MuJoCo + ONNX 推理上增加一个高层控制层。大模型只能调用 `stand`、`walk`、`turn`、`stop` 和只读的 `get_status`，不能直接控制关节。

当前默认采用双策略路由：官方 `BEST_WALK_ONNX_2.onnx` 负责 `stand`、`walk` 和 `stop`，本项目训练的 `open_duck_turn_v1.onnx` 负责 `turn`。在动作切换时会自动选择策略，`stop` 使用前一动作的策略完成平稳收尾。转向模型通过环境变量指定：

```bash
export OPEN_DUCK_ROOT="${OPEN_DUCK_ROOT:-$HOME/open_duck}"
export OPEN_DUCK_TURN_ONNX_MODEL="$OPEN_DUCK_ROOT/models/control/open_duck_turn_v1.onnx"
```

## 安装到服务器

把整个 `open_duck_agent` 目录放到：

`$OPEN_DUCK_ROOT/projects/Open_Duck_Playground/open_duck_agent`

把大模型客户端安装到 `/data` 下的独立目录。这样不会升级或覆盖已经验证过的 JAX、MuJoCo 和 ONNX Runtime：

```bash
mkdir -p "$OPEN_DUCK_ROOT/python/openai-client"
UV_HTTP_TIMEOUT=600 "$OPEN_DUCK_ROOT/bin/uv" pip install \
  --target "$OPEN_DUCK_ROOT/python/openai-client" \
  --cache-dir "$OPEN_DUCK_ROOT/cache/uv" \
  "openai>=1.68.0,<3"
```

## 第一步：不接大模型，验证四个动作

```bash
source "$OPEN_DUCK_ROOT/env_walk.sh"
cd "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  .venv/bin/python -m open_duck_agent.agent --demo
```

它会依次执行：站立 1 秒、以 0.10 m/s 行走 2 秒、以 0.20 rad/s 转向 2 秒、停止。每个动作返回位置、局部速度、竖直方向和是否摔倒。日志保存在 `$OPEN_DUCK_ROOT/outputs/agent_*/actions.jsonl`。

## 第二步：接入本地 MiniCPM-o 4.5（默认）

模型规划环境与仿真环境分离：MiniCPM-o 使用 Python 3.10 和 GPU，Open Duck 继续使用现有 Python 3.11 原生 MuJoCo 环境。默认模型目录为：

`$OPEN_DUCK_ROOT/models/MiniCPM-o-4_5-awq`

单次自然语言端到端测试：

```bash
source "$OPEN_DUCK_ROOT/env_walk.sh"
cd "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  .venv/bin/python -m open_duck_agent.agent \
  --backend minicpmo \
  --minicpmo-gpu 7 \
  --prompt "先站立1秒，以0.1米每秒前进2秒，向左转1秒，最后停止"
```

大模型输出不会直接送入关节控制。程序会再次校验动作白名单、速度、时长、动作数量并强制在末尾停止。模型规划失败或仿真检测到跌倒时同样会发送零运动命令。

## 可选：接入 OpenAI 模型

不要把 API Key 写进源码或聊天记录。在服务器终端中隐藏输入：

```bash
read -rsp "OPENAI_API_KEY: " OPENAI_API_KEY
echo
export OPENAI_API_KEY
export OPENAI_MODEL=gpt-5-mini
```

如果服务器访问外网需要现有代理，先执行：

```bash
set -a
source /path/to/proxy.env
set +a
```

单次自然语言测试：

```bash
source "$OPEN_DUCK_ROOT/env_walk.sh"
cd "$OPEN_DUCK_ROOT/projects/Open_Duck_Playground"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  .venv/bin/python -m open_duck_agent.agent \
  --backend openai \
  --prompt "先站稳，然后以0.1米每秒前进2秒，向正方向转弯1秒，最后停下"
```

不加 `--prompt` 会进入连续对话模式。输入 `quit` 退出。

## 当前安全边界

- 行走命令被限制在 -0.15 到 0.15 m/s。
- 转向命令被限制在 -0.3 到 0.3 rad/s。
- 单次动作不超过 5 秒，一次自然语言请求最多 8 个工具调用。
- `up_vector_z < 0.5`、机身高度低于 0.08 m 或状态出现非有限数时立即中断动作。
- 独立转向策略已验证正负方向均可响应；仍应使用短时动作并在每次动作后检查状态。
- 这版只控制仿真，不连接实体舵机。
