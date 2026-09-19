"""Render the collected GitHub Actions accuracy results (no network required)."""

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))
BASELINE = 96.74
LOWER, UPPER = 95.74, 97.74
rows = json.loads((ROOT / "evidence.json").read_text(encoding="utf-8"))
for row in rows:
    row["time"] = datetime.fromisoformat(row["job"]["started_at"]).astimezone(TZ)
    row["score"] = float(row["scores"][0]) if row["scores"] else None
scored = [r for r in rows if r["score"] is not None]
missing = [r for r in rows if r["score"] is None]
values = [r["score"] for r in scored]
failed = [r for r in scored if not LOWER <= r["score"] <= UPPER]

plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False})
fig = plt.figure(figsize=(15, 9), facecolor="#f8fafc")
grid = fig.add_gridspec(3, 1, height_ratios=[3, 0.42, 2.1], hspace=0.36)
ax = fig.add_subplot(grid[0])
strip = fig.add_subplot(grid[1], sharex=ax)
zoom = fig.add_subplot(grid[2])
for panel in (ax, zoom):
    panel.set_facecolor("#f8fafc")
    panel.axhspan(LOWER, UPPER, color="#dcfce7", alpha=0.65)
    panel.axhline(BASELINE, color="#64748b", linestyle="--", lw=1.2, label="基线 96.74")
    panel.axhline(LOWER, color="#ef4444", linestyle="--", lw=1.2, label="合格下限 95.74")
    subset = scored if panel is ax else [r for r in scored if r["time"].month == 9]
    panel.plot([r["time"] for r in subset], [r["score"] for r in subset],
               color="#2563eb", marker="o", markersize=6, lw=1.8, label="GSM8K 得分")
    bad = [r for r in subset if r in failed]
    panel.scatter([r["time"] for r in bad], [r["score"] for r in bad],
                  color="#dc2626", s=75, zorder=5)
    panel.grid(axis="y", alpha=0.15)
    panel.set_ylabel("Accuracy (%)")
    panel.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d", tz=TZ))
ax.set_xlim(datetime(2026, 8, 1, tzinfo=TZ), datetime(2026, 9, 20, tzinfo=TZ))
ax.set_ylim(91.5, 98.1)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5, tz=TZ))
ax.legend(loc="lower right", frameon=False, ncol=3)
ax.annotate("08-09  92.12\n低于下限 3.62 pp", (scored[0]["time"], scored[0]["score"]),
            xytext=(20, 12), textcoords="offset points", color="#b91c1c", fontsize=11)
ax.set_title("QWEN3_235B_PD · GSM8K 精度趋势", loc="left", fontsize=20, fontweight="bold", pad=30)
fig.text(0.125, 0.907, "2026-08-01 至 2026-09-19（查询时） · 北京时间 / 任务开始时间 · Weekly + Nightly",
         fontsize=11, color="#475569")
strip.set_facecolor("#f8fafc")
strip.scatter([r["time"] for r in missing], [0] * len(missing), marker="x", color="#94a3b8", s=50)
strip.set_yticks([0], ["未出分"])
strip.set_ylim(-0.8, 0.8)
strip.tick_params(axis="x", labelbottom=False, length=0)
strip.tick_params(axis="y", length=0)
zoom.set_title("9 月放大视图", loc="left", fontsize=13, fontweight="bold")
zoom.set_ylim(95.35, 96.98)
zoom.set_xlim(datetime(2026, 9, 4, tzinfo=TZ), datetime(2026, 9, 20, tzinfo=TZ))
zoom.xaxis.set_major_locator(mdates.DayLocator(interval=1, tz=TZ))
sep = [r for r in scored if r["time"].month == 9]
for i, row in enumerate(sep):
    offset = (0, 12)
    if i in (8, 10):
        offset = (0, -22)
    zoom.annotate(f"{row['score']:.2f}", (row["time"], row["score"]),
                  xytext=offset, textcoords="offset points", ha="center", fontsize=9,
                  color="#b91c1c" if row in failed else "#1e40af")
fig.text(0.125, 0.045,
         f"28 次执行 · 16 次出分 · 14 次精度通过 / 2 次不达标 · 12 次未出分 · 得分均值 {statistics.mean(values):.2f}",
         fontsize=12, fontweight="bold")
fig.text(0.125, 0.02, "折线连接有得分的运行，不代表空档期有测量；未出分不计为 0。9 月 7 日由 Weekly 迁移至 Nightly。",
         fontsize=10, color="#64748b")
fig.savefig(ROOT / "accuracy-trend.png", dpi=180, bbox_inches="tight")
fig.savefig(ROOT / "accuracy-trend.svg", bbox_inches="tight")

reasons = {
    91414604420: "Prefill 进程就绪前退出",
    95036086307: "部署文件 ./lws.yaml 不存在",
    97101945400: "Pod 就绪超时（1200 秒）",
    98737418489: "Pod 就绪超时（1200 秒）",
    101803481648: "VllmConfig 缺少 _get_v1_model_runner_unsupported_features",
    103329219873: "部署 YAML 解析错误",
    104818320953: "Prefill 进程就绪前退出",
    104827850294: "Decode ranks 就绪超时",
    105275634556: "运行取消",
    105314153314: "AISBench 预热失败：Bad Gateway",
    105506552052: "Pod 就绪超时（1201 秒）",
    105515605423: "CircularBufferSpec 导入失败",
}
lines = ["# QWEN3_235B_PD 精度运行统计", "",
         "范围：北京时间 2026-08-01 00:00 至 2026-09-19 查询时。按 job.started_at 归属日期。",
         "包含 Weekly-A3 与 Nightly-A3 的实际执行及手动触发；排除 skipped 和另一个 _3_5K_1_5k 性能用例。",
         "通过 GitHub Actions API 按周分页列举 816 个 workflow runs（含边界回溯），筛选精确配置名；复查重跑 attempts。",
         "所有 28 条实际执行日志均可取得。原始精确得分来自 AISBench Task 的 accuracy 字段，表格四舍五入至两位。", "",
         "精度口径为 GSM8K 0-shot CoT / thinking；baseline=96.74，threshold=1，合格闭区间 [95.74,97.74]。",
         "配置迁移与服务设置变化参见 [配置历史核查](config-history.md)。服务版本变化，因此不能视作同一环境的重复测量。", "",
         f"28 次执行，16 次出分：14 次通过、2 次失败；12 次未出分。有效结果通过率 87.5%。均值 {statistics.mean(values):.2f}，最小 {min(values):.2f}，最大 {max(values):.2f}。",
         "8 月 9 日 92.12 低于下限约 3.62 个百分点；8 月 30 日已恢复为 95.98。9 月 17 日 95.53 再次低于下限约 0.21 个百分点；随后三次恢复至 96.29、96.59、96.29。",
         "全部有效得分均低于基线 96.74。未出分及没有执行的日期不按零分统计，不推断低分原因。", "",
         "![精度趋势](accuracy-trend.png)", "", "## 全部执行记录", "",
         "| 开始时间（北京时间） | 工作流 | GSM8K (%) | 精度结果 / 未出分原因 | 测试 ref | GitHub 日志 |",
         "|---|---|---:|---|---|---|"]
for row in rows:
    job = row["job"]
    score = "—" if row["score"] is None else f"{row['score']:.2f}"
    result = reasons[job["id"]] if row["score"] is None else ("不达标" if row in failed else "通过")
    lines.append(f"| {row['time']:%m-%d %H:%M} | {row['run']['name']} | {score} | {result} | `{row['ref'][:12]}` | [job {job['id']}]({job['html_url']}) |")
lines += ["", "数据： [evidence.json](evidence.json)；绘图脚本：[plot_accuracy.py](plot_accuracy.py)。", ""]
(ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"runs": len(rows), "scored": len(scored), "failed": len(failed),
                  "no_score": len(missing), "mean": statistics.mean(values)}))
