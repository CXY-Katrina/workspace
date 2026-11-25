#!/usr/bin/env bash
set -euxo pipefail

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

export TMPDIR=/data/zfx/tmp
# ========== 校验当前conda环境为cy ==========
EXPECT_ENV="cy"
if [[ "${CONDA_DEFAULT_ENV}" != "${EXPECT_ENV}" ]]; then
    echo "ERROR: 当前conda环境不是【${EXPECT_ENV}】"
    echo "当前环境: ${CONDA_DEFAULT_ENV}"
    echo "请先手动 conda activate ${EXPECT_ENV} 再执行脚本"
    exit 1
fi
echo "✅ Conda环境校验通过，当前环境: ${CONDA_DEFAULT_ENV}"

pip install setuptools_rust

SOURCE_DIR=/data/zfx/code/HCF-vLLM-Hetero

cd "${SOURCE_DIR}"

pip install https://mirrors.tencent.com/repository/generic/pdtransfer/build/tag/ucl-r03c05/ascend-mc-v0.3.12.post1/nixl-1.0.0+ucl.r03c05.ascend25.7.rc1.6.0fc67343.ucx346c06f1.mc130d0917-cp310-cp310-manylinux_2_39_x86_64.whl

pip install https://mirrors.tencent.com/repository/generic/ascend-image-build/hetero-ascend-ops/master/c8b5339d-20261117_221209/8705434fff064c7caa05e8ee0b0af2ae/hetero_ascend_ops-0.0.1.dev0-cp310-cp310-linux_x86_64.whl --force-reinstall

pip install --force-reinstall --no-deps "http://mirrors.tencent.com/repository/generic/transformers/transformers-5.10.2.post4-py3-none-any.whl#sha256=c0018de0ca2eb511c84ecc907c226a43ae8435e49ee1e652e8ded9a3b316b3ad"

pip install "huggingface_hub==1.32.0" "tokenizers==0.22.2" "xgrammar==0.2.7"

pip install "openai==3.16.2"

VLLM_TARGET_DEVICE=empty pip install -v -e . --no-deps --no-build-isolation

# 临时绕过ABI检查 # 或改成 "torch_npu": "2.13.0"
mv /root/miniconda3/envs/cy/lib/python3.10/site-packages/hetero_ascend_ops/_abi.json{,.bak}


echo "===== 全部安装完成 ====="
