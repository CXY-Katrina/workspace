# A5：1P + 双机 D（DP4 TP4 EP16）

2026-10-27 实测部署快照。一个 P 实例使用一台 8 卡服务器，D 实例使用两台服务器共 16 卡。
六个 `launch.sh` / `serve.sh` 从本次采集容器的宿主机挂载目录原样导出，
校验值见 [source-manifest.json](source-manifest.json)。
[start_proxy.sh](start_proxy.sh) 对应实测代理命令。

## 拓扑与环境

| 角色 | 管理地址 | 计算网地址 | 脚本目录 | 拓扑 |
| --- | --- | --- | --- | --- |
| P | 177.11.27.1 | 197.13.51.1 | prefill | DP1 TP4 PP2 |
| D 主节点 | 177.11.28.1 | 197.13.52.1 | decode-head | local DP2，起始 rank 0 |
| D 从节点 | 177.11.21.1 | 197.13.45.1 | decode-worker | local DP2，起始 rank 2，headless |

D 全局为 DP4 TP4 PP1 EP16；每台使用 NPU 0–7。网卡为 `enp39s0f0`。

镜像：

```text
quay.io/atlas-ci/vllm-atlas-temp:minimax-m3.1-flash-a5-openeuler-40260585905-1-arm64-temp
```

本次 Docker image ID：

```text
sha256:6a51b3f0314b860a970695de96dc534035f39c1e9ee3996bb48eb5859e68208b
```

镜像内环境为 torch_npu 2.10.0.post7、CANN 9.2.0。
模型为 MiniMax M3.1 Flash quarot l32，容器内权重路径固定为：

```text
/data/l00845740/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32
```

权重和镜像需预先准备。实际采集时权重从 26.1 的只读 NFS 挂载，
该临时挂载和 export 已在测试后清理。重新部署时必须提供可用的模型目录；
不要直接把已卸载后的空目录当作权重。

## 关键参数

| 参数 | P | D |
| --- | --- | --- |
| Prefix cache | 开启 | 开启 |
| DSpark | 7 个草稿 token | 7 个草稿 token |
| 主模型 / KV 精度 | bf16 + ascend 量化 / fp8 | bf16 + ascend 量化 / fp8 |
| DSpark KV | bf16 | bf16 |
| max model len | 133000 | 133000 |
| max num seqs | 128 | 每个 DP engine 256 |
| max batched tokens | 16384 | 32768 |
| memory utilization | 0.85 | 0.92 |
| HCCL_BUFFSIZE | 512 | 1024 |
| 执行模式 | enforce eager | FULL_DECODE_ONLY，max cudagraph capture size 1024 |

D 使用 `fuse_norm_quant=false`、`fuse_qknorm_rope=true`。
两端通过 `MooncakeConnectorV1` + `use_ascend_direct=true` 传输 KV。
D 两台必须使用相同权重、镜像及关键配置。

D 保留本次 `--profiler-config`，目录为
`/work/decode-profile-20261027/profiling`。仅在收到 `/start_profile` 后开始采集。

## 准备容器

先检查 `npu-smi info`、现有容器及端口。仅使用已分配且空闲的资源，不停止其他用户任务。
下面的命令是从实测挂载整理的创建示例；需要在每台机器分别执行，并配置该机器的角色、
宿主机工作目录和实际权重目录。创建同名容器时 Docker 会报错，不会替换现有容器。

```bash
# 在本部署目录下执行。三台分别使用：
set -e
# ROLE=prefill       NAME=decode-a5-prefill
# ROLE=decode-head   NAME=decode-a5-head
# ROLE=decode-worker NAME=decode-a5-worker
ROLE=prefill
NAME=decode-a5-prefill
SCRIPTS_DIR="$(pwd)"
SESSION_DIR=/data/a5-pd-20261027
MODEL_HOST_PATH=/path/to/prepared/model
IMAGE=quay.io/atlas-ci/vllm-atlas-temp:minimax-m3.1-flash-a5-openeuler-40260585905-1-arm64-temp

test -f "$MODEL_HOST_PATH/config.json"
test -f "$SCRIPTS_DIR/$ROLE/serve.sh"
mkdir -p "$SESSION_DIR"/{service,profiling,cache,tmp,ascend}
devices=()
for device in /dev/davinci{0..7} /dev/davinci_manager /dev/hisi_hdc; do
    test -e "$device" || exit 1
    devices+=(--device "$device")
done

docker run -d --name "$NAME" \
    --label bench.session=a5-pd-20261027 \
    --network host --privileged --shm-size 100g \
    --add-host decode-profile-p:197.13.51.1 \
    --add-host decode-profile-d0:197.13.52.1 \
    --add-host decode-profile-d1:197.13.45.1 \
    "${devices[@]}" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /etc/hixlep:/etc/hixlep:ro \
    -v /etc/hccn.conf:/etc/hccn.conf:ro \
    -v "$MODEL_HOST_PATH":/data/l00845740/minimax_m3dot1_flash_w8a8_mxfp8_quarot_l32:ro \
    -v "$SESSION_DIR":/work/decode-profile-20261027 \
    -v "$SCRIPTS_DIR/$ROLE":/work/decode-profile-20261027/config:ro \
    -v "$SCRIPTS_DIR/start_proxy.sh":/work/decode-profile-20261027/start_proxy.sh:ro \
    -v "$SESSION_DIR/cache":/root/.cache \
    -v "$SESSION_DIR/tmp":/tmp \
    -v "$SESSION_DIR/ascend":/root/ascend \
    --entrypoint sleep "$IMAGE" infinity
```

三条 hosts 映射用于计算网络地址解析。本次部署中缺少这些映射时，
TCPStore 的反向 DNS 查询曾使三机初始化长时间等待；加入映射后 16 rank Gloo 检查通过。
更换服务器时同步修改 hosts、各角色脚本中的 IP、网卡，以及 proxy 的地址。

实测 21.1 使用位于 `/mnt/weight/decode-profile-20261027` 的私有 Docker daemon，
该 daemon 已停止。上述示例默认使用当前 Docker endpoint；
如使用其他 endpoint，须对同一容器的所有命令保持一致，不要启停其他人的 daemon。

## 启动与连通性验证

两台 D 需要配套启动，不要等待 D 主节点健康后才启动从节点。
以下命令在各自宿主机执行，并写入各自挂载目录。

```bash
# 27.1：P
docker exec -d decode-a5-prefill bash -lc \
    'bash /work/decode-profile-20261027/config/launch.sh > /work/decode-profile-20261027/service/service.log 2>&1'

# 28.1：D 主节点
docker exec -d decode-a5-head bash -lc \
    'bash /work/decode-profile-20261027/config/launch.sh > /work/decode-profile-20261027/service/service.log 2>&1'

# 21.1：D 从节点，紧接着启动
docker exec -d decode-a5-worker bash -lc \
    'bash /work/decode-profile-20261027/config/launch.sh > /work/decode-profile-20261027/service/service.log 2>&1'
```

镜像中的 `CANN_VERSION` 供原始 `launch.sh` 加载对应运行环境。
日志出现健康就绪后，从实际客户端所在网络检查：

```bash
curl --noproxy '*' -fsS http://197.13.51.1:31150/health
curl --noproxy '*' -fsS http://197.13.52.1:31160/health
curl --noproxy '*' -fsS http://197.13.52.1:31160/v1/models

# 27.1：P/D 均就绪后启动代理
docker exec -d decode-a5-prefill bash -lc \
    'bash /work/decode-profile-20261027/start_proxy.sh > /work/decode-profile-20261027/service/proxy.log 2>&1'
```

完整 PD 请求入口为 `http://197.13.51.1:8019/v1/chat/completions`，
模型名为 `minimax-m3.1-flash`。代理使用 4 workers、max retries 0。
首次真实请求可能触发编译，应先单独完成预热。
D 从节点为 headless，不提供独立 HTTP API。

固定端口：P API 31150 / RPC 6984 / KV 30100 / proxy 8019；
D 主节点 API 31160 / RPC 6985；两台 D KV 27900。
HCCL/Gloo/Mooncake 还会使用动态连接，计算网络必须互通。

## 本次验证范围

- 16K/1K、BS230：完整 PD，P/D 开启 prefix cache，prefix_num=1、100% 重复；
  920/920 请求成功，输出均为 1024 tokens。块级 prefix 命中率为 98.4375%，
  剩余 256 tokens 由 P 计算并传给 D。
- 128K/1K、BS60：相同单前缀配置；各 DP 预热输入均为 131072 tokens。
  running 达到 60 后采集 3 秒，随后按要求终止 AISBench，不作为完整性能成绩。
- 两组均取得 D 的 16 rank 原始 profiling，并完成 trace 与算子表离线解析。
- 所有 k 均为 1024。BS60 的本次验证采用共享单前缀，不代表 60 个独立上下文的 KV 容量。
- 以上是已执行的服务配置；此次仓库同步仅做脚本和参数校验，未重新启动服务。

## 停止本次部署

在各自宿主机，仅停止本次创建且确认归属的容器：

```bash
# 27.1（同时停止本容器内的 proxy）
docker stop --time 20 decode-a5-prefill
# 28.1
docker stop --time 20 decode-a5-head
# 21.1
docker stop --time 20 decode-a5-worker
```

停止后检查对应容器 Running=false、Pid=0 和对应服务端口关闭。
本目录不含服务器登录凭据；profiling 大文件不放入 Git。

同步检查：7 个 shell 脚本均通过 `bash -n`；6 个原始文件 SHA-256 与服务器导出一致；
通过模拟 `vllm` 命令检查了三种角色的实际展开参数、JSON 配置、DP rank 和 graph size。
