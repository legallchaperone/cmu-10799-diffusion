import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import shutil
    import signal
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    return Path, mo, os, shutil, signal, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # DDPM training on molab

    训练以**独立后台进程**运行（`start_new_session=True`），输出写到 `train.out`。
    浏览器断开、notebook 重连、甚至 kernel 重启都不会杀掉训练；重新打开 notebook 后，
    下面的状态面板会自动接上。

    步骤：① 配置 → ② Setup（clone / pull + 装依赖）→ ③ Start → ④ 看状态面板。
    """)
    return


@app.cell
def _(Path):
    # ① 配置：按需修改
    REPO_URL = "https://github.com/legallchaperone/cmu-10799-diffusion.git"
    # 如果当前目录（或上一级）已经是仓库就直接用，否则 clone 到 ./cmu-10799-diffusion
    WORK = next(
        (d for d in (Path.cwd(), Path.cwd().parent) if (d / "train.py").exists()),
        Path.cwd() / "cmu-10799-diffusion",
    )
    CONFIG = "configs/ddpm_molab.yaml"
    LOG_FILE = WORK / "train.out"
    PID_FILE = WORK / "train.pid"
    return CONFIG, LOG_FILE, PID_FILE, REPO_URL, WORK


@app.cell
def _(LOG_FILE, PID_FILE, WORK, os):
    def running_pid():
        """训练进程的 pid；没在跑则返回 None。"""
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # 只检查进程是否存在，不发信号
            return pid
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            return None

    def latest(pattern):
        files = sorted(WORK.glob(pattern), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    def tail_log(n=25):
        if not LOG_FILE.exists():
            return ["(还没有日志)"]
        text = LOG_FILE.read_text(errors="replace")[-200_000:]
        # tqdm 用 \r 原地刷新进度条，只保留每行最后一次刷新
        lines = [ln.split("\r")[-1] for ln in text.split("\n")]
        return [ln for ln in lines if ln.strip()][-n:]

    return latest, running_pid, tail_log


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="② Setup: clone/pull + 安装依赖")
    wandb_key = mo.ui.text(kind="password", label="W&B API key（留空 = 不用 wandb）", full_width=True)
    mo.vstack([setup_btn, wandb_key])
    return setup_btn, wandb_key


@app.cell
def _(REPO_URL, WORK, mo, setup_btn, shutil, subprocess, sys):
    mo.stop(not setup_btn.value)

    def _run(cmd, cwd=None):
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return f"$ {' '.join(map(str, cmd))}\n{r.stdout[-3000:]}{r.stderr[-3000:]}"

    _out = []
    if (WORK / ".git").exists():
        _out.append(_run(["git", "pull", "--ff-only"], cwd=WORK))
    else:
        _out.append(_run(["git", "clone", REPO_URL, str(WORK)]))

    _deps = ["-e", ".[wandb,datasets]"]
    if shutil.which("uv"):
        _out.append(_run(["uv", "pip", "install", "--python", sys.executable, *_deps], cwd=WORK))
    else:
        _out.append(_run([sys.executable, "-m", "pip", "install", *_deps], cwd=WORK))

    mo.md("```\n" + "\n\n".join(_out) + "\n```")
    return


@app.cell
def _(mo):
    resume_box = mo.ui.checkbox(value=True, label="从最新 checkpoint 续训（没有就从头开始）")
    start_btn = mo.ui.run_button(label="③ Start training", kind="success")
    stop_btn = mo.ui.run_button(label="Stop training", kind="danger")
    mo.hstack([start_btn, resume_box, stop_btn], justify="start")
    return resume_box, start_btn, stop_btn


@app.cell
def _(
    CONFIG,
    LOG_FILE,
    PID_FILE,
    WORK,
    latest,
    mo,
    os,
    resume_box,
    running_pid,
    start_btn,
    subprocess,
    sys,
    wandb_key,
):
    mo.stop(not start_btn.value)

    if not (WORK / "train.py").exists():
        _msg = f"`{WORK}` 里还没有代码：先点 **② Setup**，确认输出里 git clone 成功，再点 Start。"
    elif running_pid() is not None:
        _msg = f"已经在训练了（pid {running_pid()}），没有重复启动。"
    else:
        _cmd = [sys.executable, "-u", "train.py", "--method", "ddpm", "--config", CONFIG]
        _ckpt = latest("logs/*/checkpoints/*.pt") if resume_box.value else None
        if _ckpt is not None:
            _cmd += ["--resume", str(_ckpt)]

        # C.UTF-8 一定存在；molab 镜像没装 en_US.UTF-8，会刷 setlocale 警告
        _env = dict(os.environ, PYTHONUNBUFFERED="1", LC_ALL="C.UTF-8", LANG="C.UTF-8")
        if wandb_key.value.strip():
            _env["WANDB_API_KEY"] = wandb_key.value.strip()
        else:
            _env["WANDB_MODE"] = "disabled"  # 没 key 时 wandb.init 不会卡住等输入

        with open(LOG_FILE, "a") as _f:
            _f.write(f"\n===== launch: {' '.join(_cmd)} =====\n")
            _proc = subprocess.Popen(
                _cmd,
                cwd=WORK,
                stdout=_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=_env,
                start_new_session=True,  # 脱离 kernel：kernel 断了训练也不停
            )
        PID_FILE.write_text(str(_proc.pid))
        _msg = f"已启动，pid {_proc.pid}" + (f"，从 `{_ckpt.relative_to(WORK)}` 续训" if _ckpt else "，从头训练")
    mo.callout(mo.md(_msg), kind="info")
    return


@app.cell
def _(mo, os, running_pid, signal, stop_btn):
    mo.stop(not stop_btn.value)
    _pid = running_pid()
    if _pid is None:
        _msg = "没有正在运行的训练。"
    else:
        os.killpg(_pid, signal.SIGTERM)
        _msg = f"已发送停止信号给 pid {_pid}。之后可勾选续训再 Start。"
    mo.callout(mo.md(_msg), kind="warn")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["10s", "30s", "1m"], default_interval="30s")
    refresh
    return (refresh,)


@app.cell
def _(WORK, latest, mo, refresh, running_pid, tail_log):
    refresh  # 依赖 refresh，让本 cell 定时重跑

    _pid = running_pid()
    _status = f"🟢 训练中（pid {_pid}）" if _pid else "⚪️ 没有训练在运行"
    _ckpt = latest("logs/*/checkpoints/*.pt")
    _sample = latest("logs/*/samples/*.png")

    _items = [
        mo.md(f"**{_status}**  \n最新 checkpoint：`{_ckpt.relative_to(WORK) if _ckpt else '无'}`"),
        mo.md("```\n" + "\n".join(tail_log()) + "\n```"),
    ]
    if _sample is not None:
        _items.append(mo.md(f"最新样本：`{_sample.relative_to(WORK)}`"))
        _items.append(mo.image(src=_sample.read_bytes(), width=420))
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
