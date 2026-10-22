可以，只测前几条，直接在原命令后加 --num-prompts N。我核对了 AISBench 当前 master 的文档和实现，该参数也适用于精度测评。
例如，只测试 MMMU-Pro 的前 5 条：
ais_bench --models vllm_api_general_chat \
  --datasets mmmu_pro_options10_gen \
  --num-prompts 5
将模型和数据集名称换成你当前使用的配置即可；mmmu_pro_options10_cot_gen、mmmu_pro_vision_gen、mmmu_pro_vision_cot_gen 同样适用。使用自定义配置文件时：
ais_bench your_config.py --num-prompts 5
该参数按数据集顺序取前 N 条。官方参数说明
如果要选择某一段数据，在自定义配置中设置 reader_cfg.test_range：
# 放在导入 datasets 之后，保留原有模型等配置
datasets[0]['reader_cfg']['test_range'] = '[10:15]'
这里是从 0 开始的索引，左闭右开，即只测第 11～15 条。也可以直接修改所用 MMMU-Pro 数据集配置里的读取器：
mmmu_pro_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer',
    test_range='[0:5]',  # 前 5 条
)
注意：
- test_range 的切片写法要带引号。
- 同时设置时，命令行 --num-prompts 会覆盖 test_range。
- 若旧版本不识别 --num-prompts，可使用上述 test_range 配置方式。
参考：MMMU-Pro 配置源码、范围读取实现。
