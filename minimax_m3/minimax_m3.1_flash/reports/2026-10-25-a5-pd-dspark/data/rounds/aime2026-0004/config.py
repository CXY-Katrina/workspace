from ais_bench.benchmark.models import VLLMCustomAPIChat
try:
    from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
except ImportError:
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [dict(
    attr='service', type=VLLMCustomAPIChat, abbr='minimax-pd-dspark',
    path='', model='minimax-m3.1-flash', host_ip='177.11.26.1', host_port=8009,
    stream=True, request_rate=0, retry=1, api_key='',
    max_out_len=81920, batch_size=16,
    generation_kwargs=dict(temperature=1, top_p=0.95, seed=45, chat_template_kwargs=dict(reasoning_effort='max', thinking_mode='enabled'), ignore_eos=False),
    pred_postprocessor=dict(type=extract_non_reasoning_content),
)]

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import (
    Aime2026Dataset,
    MATHEvaluator,
    math_postprocess_v2,
)

aime2026_reader_cfg = dict(input_columns=["problem"], output_column="answer")

aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="\nSolve the following math problem step by step. Put your answer inside \\boxed{}.\n\n{problem}\n\nRemember to put your answer inside \\boxed{}.",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aime2026_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version="v2"),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

aime2026_datasets = [
    dict(
        abbr="aime2026",
        type=Aime2026Dataset,
        path="ais_bench/datasets/aime2026/aime2026.jsonl",
        reader_cfg=aime2026_reader_cfg,
        infer_cfg=aime2026_infer_cfg,
        eval_cfg=aime2026_eval_cfg,
    )
]

aime2026_datasets[0]['path'] = '/home/j00893003/minimax-dspark/cxy-pd-pr19723-20261025/aime2026.jsonl'
datasets = aime2026_datasets
summarizer = dict(attr='accuracy', summary_groups=[])
