# 无 DSpark GPQA 第二轮已返回130题的完整记录

第二轮使用seed1043，在130/198个请求返回时按用户要求中止。本目录保存这130条成功返回的全部原始回答，包括临时核对正确和错误的回答。

**已返回子集临时核对：114/130匹配参考选项，87.69%；16题未匹配。** 仍有68题未获得返回，因此不能将87.69%视为完整198题的正式精度，也不将本轮合并进第一轮165/198（83.33%）的正式成绩。

答案先按AISBench的 `extract_non_reasoning_content` 规则取非推理内容，再按 `GPQA_Simple_Eval_postprocess` 提取选项，与每条返回记录的gold比较。GPQA题号为CSV行号加1，Record ID及该次请求选项随原始回答保留。

暂未匹配的题号：9、11、18、22、28、36、45、64、70、76、86、93、95、103、107、114。

[全部130条原始JSONL压缩包](returned-records.jsonl.gz) / [结构化逐题核对](per-question.json) / [子集统计](summary.json) / [中止记录](user-stopped-partial-round.json)。

| ID | 临时提取答案 | 该次参考选项 | 核对 | 记录 |
| --- | --- | --- | --- | --- |
| 1 | D | D | 匹配 | [完整回答](answers/ID001.md) |
| 2 | C | C | 匹配 | [完整回答](answers/ID002.md) |
| 3 | B | B | 匹配 | [完整回答](answers/ID003.md) |
| 4 | A | A | 匹配 | [完整回答](answers/ID004.md) |
| 5 | D | D | 匹配 | [完整回答](answers/ID005.md) |
| 6 | C | C | 匹配 | [完整回答](answers/ID006.md) |
| 7 | B | B | 匹配 | [完整回答](answers/ID007.md) |
| 8 | A | A | 匹配 | [完整回答](answers/ID008.md) |
| 9 | A | D | 未匹配 | [完整回答](answers/ID009.md) |
| 10 | C | C | 匹配 | [完整回答](answers/ID010.md) |
| 11 | D | B | 未匹配 | [完整回答](answers/ID011.md) |
| 12 | A | A | 匹配 | [完整回答](answers/ID012.md) |
| 13 | D | D | 匹配 | [完整回答](answers/ID013.md) |
| 14 | C | C | 匹配 | [完整回答](answers/ID014.md) |
| 15 | B | B | 匹配 | [完整回答](answers/ID015.md) |
| 16 | A | A | 匹配 | [完整回答](answers/ID016.md) |
| 17 | D | D | 匹配 | [完整回答](answers/ID017.md) |
| 18 | D | C | 未匹配 | [完整回答](answers/ID018.md) |
| 19 | B | B | 匹配 | [完整回答](answers/ID019.md) |
| 20 | A | A | 匹配 | [完整回答](answers/ID020.md) |
| 21 | D | D | 匹配 | [完整回答](answers/ID021.md) |
| 22 | A | C | 未匹配 | [完整回答](answers/ID022.md) |
| 23 | B | B | 匹配 | [完整回答](answers/ID023.md) |
| 24 | A | A | 匹配 | [完整回答](answers/ID024.md) |
| 25 | D | D | 匹配 | [完整回答](answers/ID025.md) |
| 26 | C | C | 匹配 | [完整回答](answers/ID026.md) |
| 27 | B | B | 匹配 | [完整回答](answers/ID027.md) |
| 28 | C | A | 未匹配 | [完整回答](answers/ID028.md) |
| 29 | D | D | 匹配 | [完整回答](answers/ID029.md) |
| 30 | C | C | 匹配 | [完整回答](answers/ID030.md) |
| 31 | B | B | 匹配 | [完整回答](answers/ID031.md) |
| 32 | A | A | 匹配 | [完整回答](answers/ID032.md) |
| 33 | D | D | 匹配 | [完整回答](answers/ID033.md) |
| 34 | C | C | 匹配 | [完整回答](answers/ID034.md) |
| 35 | B | B | 匹配 | [完整回答](answers/ID035.md) |
| 36 | C | A | 未匹配 | [完整回答](answers/ID036.md) |
| 37 | D | D | 匹配 | [完整回答](answers/ID037.md) |
| 38 | C | C | 匹配 | [完整回答](answers/ID038.md) |
| 39 | B | B | 匹配 | [完整回答](answers/ID039.md) |
| 40 | A | A | 匹配 | [完整回答](answers/ID040.md) |
| 41 | D | D | 匹配 | [完整回答](answers/ID041.md) |
| 42 | C | C | 匹配 | [完整回答](answers/ID042.md) |
| 43 | B | B | 匹配 | [完整回答](answers/ID043.md) |
| 44 | A | A | 匹配 | [完整回答](answers/ID044.md) |
| 45 | B | D | 未匹配 | [完整回答](answers/ID045.md) |
| 46 | C | C | 匹配 | [完整回答](answers/ID046.md) |
| 47 | B | B | 匹配 | [完整回答](answers/ID047.md) |
| 48 | A | A | 匹配 | [完整回答](answers/ID048.md) |
| 50 | C | C | 匹配 | [完整回答](answers/ID050.md) |
| 51 | B | B | 匹配 | [完整回答](answers/ID051.md) |
| 52 | A | A | 匹配 | [完整回答](answers/ID052.md) |
| 53 | D | D | 匹配 | [完整回答](answers/ID053.md) |
| 54 | C | C | 匹配 | [完整回答](answers/ID054.md) |
| 55 | B | B | 匹配 | [完整回答](answers/ID055.md) |
| 56 | A | A | 匹配 | [完整回答](answers/ID056.md) |
| 57 | D | D | 匹配 | [完整回答](answers/ID057.md) |
| 58 | C | C | 匹配 | [完整回答](answers/ID058.md) |
| 59 | B | B | 匹配 | [完整回答](answers/ID059.md) |
| 60 | A | A | 匹配 | [完整回答](answers/ID060.md) |
| 61 | D | D | 匹配 | [完整回答](answers/ID061.md) |
| 62 | C | C | 匹配 | [完整回答](answers/ID062.md) |
| 63 | B | B | 匹配 | [完整回答](answers/ID063.md) |
| 64 | C | A | 未匹配 | [完整回答](answers/ID064.md) |
| 65 | D | D | 匹配 | [完整回答](answers/ID065.md) |
| 66 | C | C | 匹配 | [完整回答](answers/ID066.md) |
| 67 | B | B | 匹配 | [完整回答](answers/ID067.md) |
| 68 | A | A | 匹配 | [完整回答](answers/ID068.md) |
| 69 | D | D | 匹配 | [完整回答](answers/ID069.md) |
| 70 | A | C | 未匹配 | [完整回答](answers/ID070.md) |
| 71 | B | B | 匹配 | [完整回答](answers/ID071.md) |
| 73 | D | D | 匹配 | [完整回答](answers/ID073.md) |
| 74 | C | C | 匹配 | [完整回答](answers/ID074.md) |
| 75 | B | B | 匹配 | [完整回答](answers/ID075.md) |
| 76 | C | A | 未匹配 | [完整回答](answers/ID076.md) |
| 78 | C | C | 匹配 | [完整回答](answers/ID078.md) |
| 79 | B | B | 匹配 | [完整回答](answers/ID079.md) |
| 81 | D | D | 匹配 | [完整回答](answers/ID081.md) |
| 83 | B | B | 匹配 | [完整回答](answers/ID083.md) |
| 84 | A | A | 匹配 | [完整回答](answers/ID084.md) |
| 85 | D | D | 匹配 | [完整回答](answers/ID085.md) |
| 86 | B | C | 未匹配 | [完整回答](answers/ID086.md) |
| 87 | B | B | 匹配 | [完整回答](answers/ID087.md) |
| 88 | A | A | 匹配 | [完整回答](answers/ID088.md) |
| 90 | C | C | 匹配 | [完整回答](answers/ID090.md) |
| 92 | A | A | 匹配 | [完整回答](answers/ID092.md) |
| 93 | B | D | 未匹配 | [完整回答](answers/ID093.md) |
| 94 | C | C | 匹配 | [完整回答](answers/ID094.md) |
| 95 | D | B | 未匹配 | [完整回答](answers/ID095.md) |
| 96 | A | A | 匹配 | [完整回答](answers/ID096.md) |
| 97 | D | D | 匹配 | [完整回答](answers/ID097.md) |
| 98 | C | C | 匹配 | [完整回答](answers/ID098.md) |
| 99 | B | B | 匹配 | [完整回答](answers/ID099.md) |
| 100 | A | A | 匹配 | [完整回答](answers/ID100.md) |
| 101 | D | D | 匹配 | [完整回答](answers/ID101.md) |
| 102 | C | C | 匹配 | [完整回答](answers/ID102.md) |
| 103 | C | B | 未匹配 | [完整回答](answers/ID103.md) |
| 104 | A | A | 匹配 | [完整回答](answers/ID104.md) |
| 105 | D | D | 匹配 | [完整回答](answers/ID105.md) |
| 106 | C | C | 匹配 | [完整回答](answers/ID106.md) |
| 107 | D | B | 未匹配 | [完整回答](answers/ID107.md) |
| 108 | A | A | 匹配 | [完整回答](answers/ID108.md) |
| 109 | D | D | 匹配 | [完整回答](answers/ID109.md) |
| 110 | C | C | 匹配 | [完整回答](answers/ID110.md) |
| 111 | B | B | 匹配 | [完整回答](answers/ID111.md) |
| 112 | A | A | 匹配 | [完整回答](answers/ID112.md) |
| 113 | D | D | 匹配 | [完整回答](answers/ID113.md) |
| 114 | A | C | 未匹配 | [完整回答](answers/ID114.md) |
| 115 | B | B | 匹配 | [完整回答](answers/ID115.md) |
| 117 | D | D | 匹配 | [完整回答](answers/ID117.md) |
| 118 | C | C | 匹配 | [完整回答](answers/ID118.md) |
| 120 | A | A | 匹配 | [完整回答](answers/ID120.md) |
| 121 | D | D | 匹配 | [完整回答](answers/ID121.md) |
| 122 | C | C | 匹配 | [完整回答](answers/ID122.md) |
| 123 | B | B | 匹配 | [完整回答](answers/ID123.md) |
| 124 | A | A | 匹配 | [完整回答](answers/ID124.md) |
| 125 | D | D | 匹配 | [完整回答](answers/ID125.md) |
| 126 | C | C | 匹配 | [完整回答](answers/ID126.md) |
| 127 | B | B | 匹配 | [完整回答](answers/ID127.md) |
| 129 | D | D | 匹配 | [完整回答](answers/ID129.md) |
| 130 | C | C | 匹配 | [完整回答](answers/ID130.md) |
| 132 | A | A | 匹配 | [完整回答](answers/ID132.md) |
| 133 | D | D | 匹配 | [完整回答](answers/ID133.md) |
| 134 | C | C | 匹配 | [完整回答](answers/ID134.md) |
| 135 | B | B | 匹配 | [完整回答](answers/ID135.md) |
| 136 | A | A | 匹配 | [完整回答](answers/ID136.md) |
| 137 | D | D | 匹配 | [完整回答](answers/ID137.md) |
| 138 | C | C | 匹配 | [完整回答](answers/ID138.md) |
| 139 | B | B | 匹配 | [完整回答](answers/ID139.md) |
| 141 | D | D | 匹配 | [完整回答](answers/ID141.md) |
| 142 | C | C | 匹配 | [完整回答](answers/ID142.md) |

未获得返回的题号：49、72、77、80、82、89、91、116、119、128、131、140、143、144、145、146、147、148、149、150、151、152、153、154、155、156、157、158、159、160、161、162、163、164、165、166、167、168、169、170、171、172、173、174、175、176、177、178、179、180、181、182、183、184、185、186、187、188、189、190、191、192、193、194、195、196、197、198。
