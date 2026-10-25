from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GPQADataset, GPQA_Simple_Eval_postprocess, GPQAEvaluator
try:
    from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
except ImportError:
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [dict(
    attr='service', type=VLLMCustomAPIChat, abbr='minimax-pd-dspark',
    path='', model='minimax-m3.1-flash', host_ip='177.11.26.1', host_port=8009,
    stream=True, request_rate=0, retry=1, api_key='',
    max_out_len=81920, batch_size=16,
    generation_kwargs=dict(temperature=1, top_p=0.95, seed=1043, chat_template_kwargs=dict(reasoning_effort='max', thinking_mode='enabled'), ignore_eos=False),
    pred_postprocessor=dict(type=extract_non_reasoning_content),
)]
prompt = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""
datasets = [dict(
    abbr='GPQA_diamond', type=GPQADataset, path='/home/j00893003/minimax-dspark/cxy-pd-pr19723-20261025', name='gpqa_diamond.csv',
    reader_cfg=dict(input_columns=['question','A','B','C','D'], output_column='answer', test_range=None),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template=dict(round=[dict(role='HUMAN',prompt=prompt)])),
                   retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)),
    eval_cfg=dict(evaluator=dict(type=GPQAEvaluator), pred_postprocessor=dict(type=GPQA_Simple_Eval_postprocess)),
)]
summarizer = dict(attr='accuracy', summary_groups=[])
