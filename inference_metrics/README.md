# 推理系统性能指标图解

打开 [完整交互图文](index.html)。内容包含：

- Prefill → Decode 流程与 TTFT、ITL、TPOT、E2E 的起止边界。
- QPS、系统输出 TPS 的统计窗口，以及瞬时/平均并发。
- 基础观测、派生计算和均值/分位数统计的三层区分。
- 客户端与服务端、token 与输出块、目标负载与实际性能的口径差异。

[单请求范围图](single-request.png) · [系统统计窗口图](system-window.png) · [Archify 时序图](prefill-decode.html)

全部数值是教学示例。完整说明附一手来源，可离线阅读正文。

## 检查记录

主图文页通过 1440×900、1600×1000、1920×1080、2048×1320 和 390×844 的横向包含检查；长文按设计纵向滚动。测量范围高亮和并发滑块已验证，无页面脚本错误。两张范围图及分类表已进行图像复核。

Archify 时序图交付回执：

```text
diagram_type: sequence
output: C:/Users/Katrina/orca/workspaces/workspace/main-2/inference_metrics/prefill-decode.html
specification_sha256: 39400f966949ac9fa1eb817f5767616ba0b3cd498ad340687c7f3390dce4a550
artifact_sha256: 959d1cbf2b4ed724c313b56a1315f53bb6acb58038cec69476e0e9e71857bd53
specification_bytes: 2883
artifact_bytes: 705558
validation: 9/9 showcase, 0 errors, 0 warnings
browser_evidence: passed
visual_review: passed
correction_rounds: 2
```

浏览器证据使用 Edge / Chromium，覆盖四种桌面尺寸和端点尺寸的明暗主题；图像复核检查了 2048×1320 浅色与 1440×900 深色截图。自动检查记录与人工式图像复核分别报告；自动回执中的 visualReview: pending 不代表其执行了视觉审查。

源文件为 `build_guide.py` 和 `prefill-decode.sequence.json`；浏览器检查记录为 `guide-check.json` 和 `prefill-decode.visual-check.json`。
