# 游戏舆情情感分析蒸馏方案

学生模型权重：[bruceant/game-sentiment-student-v1](https://huggingface.co/bruceant/game-sentiment-student-v1)。

本仓库公开训练、评测和推理代码，供个人学习与研究。真实爬取评论、人工/教师标注语料和模型权重不在仓库里；数据库账号、API Key 只从环境变量读取，模板见 `.env.example`。仓内 `data/samples/` 只有虚构示例，不能当作训练集。

针对 B站/微博/小红书/抖音 多平台游戏相关评论、日增60万-200万量级的场景设计。
核心思路：**大模型（Qwen + DeepSeek）只负责标注训练数据，不参与生产推理；
生产环境跑轻量 BERT 分类模型，靠蒸馏数据的质量弥补模型体量的差距**。

## 整体流程

```
                 ┌─────────────────────────────────────────────┐
                 │  阶段一：冷启动                                 │
                 │  data_prep.py --mode cold_start              │
                 │  从500万存量里分层抽样 2-3万条候选样本            │
                 └───────────────────┬───────────────────────────┘
                                     ↓
                 ┌─────────────────────────────────────────────┐
                 │  阶段二：双模型交叉标注                          │
                 │  teacher_labeling.py                          │
                 │  Qwen + DeepSeek 各标一遍，只保留两者一致的结果   │
                 │  不一致的样本 -> needs_review/ 人工复核          │
                 └───────────────────┬───────────────────────────┘
                                     ↓
                 ┌─────────────────────────────────────────────┐
                 │  阶段三：学生模型微调                            │
                 │  train_student.py                             │
                 │  在 Erlangshen-Roberta-110M-Sentiment 基础上   │
                 │  微调出 student_v1                             │
                 └───────────────────┬───────────────────────────┘
                                     ↓
                 ┌─────────────────────────────────────────────┐
                 │  阶段四：生产环境批量推理                        │
                 │  infer_batch.py                               │
                 │  处理每日新增60万-200万条评论                    │
                 │  同时挖掘低置信度样本 -> 下一轮候选池             │
                 └───────────────────┬───────────────────────────┘
                                     ↓
                 ┌─────────────────────────────────────────────┐
                 │  阶段五：持续迭代（每周/每两周循环一次）           │
                 │  data_prep.py --mode active_learning         │
                 │  用低置信度样本 -> teacher_labeling.py 标注    │
                 │  -> train_student.py --init_from student_v1  │
                 │  产出 student_v2，滚动更新                      │
                 └─────────────────────────────────────────────┘
```

## 环境准备

```bash
pip install -r requirements.txt

# LLM API Key（标注阶段用，生产推理阶段不需要）
export DASHSCOPE_API_KEY="你的阿里云百炼API Key"     # Qwen
export DEEPSEEK_API_KEY="你的DeepSeek API Key"
```

## 把训练好的模型发给别人

Git 仓库**不含** `models/` 权重（`student_v1` 约 390MB）。训练完成后用 `share_student.py` 打离线包，或传到 Hugging Face，对方不必克隆本仓库。

标签：`负面` / `中性` / `正面`（`config.json` 的 `id2label`）。序列长度 256。推理自动选 `cuda` → `mps` → `cpu`。

### 打离线 zip

```bash
python share_student.py --model_dir models/student_v1
# 产出 dist/student_v1/ 和 dist/student_v1.zip
```

zip 内含：`config.json`、tokenizer、`model.safetensors`、`run_student.py`、`requirements.txt`、模型说明。不含训练 checkpoint。

对方解压后（全程可离线）：

```bash
pip install -r requirements.txt
python run_student.py --model . --text "这游戏太坑了" --local-files-only
python run_student.py --model . --input comments.txt --output pred.jsonl --local-files-only
```

`--input` 支持一行一条的 txt，或带 `text` 字段的 jsonl。也可 `echo '文本' | python run_student.py --model . --local-files-only`。

Python 直接加载：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = "."  # 解压目录
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
model.eval()
```

### 传到 Hugging Face

需先 `huggingface-cli login` 或 `export HF_TOKEN=...`。默认创建**私有**仓库，公开加 `--public`。

```bash
python share_student.py --model_dir models/student_v1 \
    --hf-repo 你的用户名/game-sentiment-student-v1
```

对方：

```bash
python run_student.py --model 你的用户名/game-sentiment-student-v1 --text "这游戏太坑了"
```

已缓存到本地后可加 `--local-files-only`。

### 本仓库里直接推理

本机已有 `models/student_v1` 时：

```bash
python run_student.py --model models/student_v1 --text "这游戏太坑了" --local-files-only
```

生产批量（CSV，需 `text` 列）仍用 `infer_batch.py`，见下方步骤 4。

## 原始数据格式

原始数据格式要求（CSV），至少包含以下列：

| 列名 | 说明 | 是否必须 |
|---|---|---|
| text | 评论正文 | 必须 |
| platform | bilibili / weibo / xiaohongshu / douyin | 建议提供，缺失则标为 unknown |
| id | 唯一标识 | 建议提供，缺失则自动生成 |
| game_category | 游戏品类（可选，用于分层抽样） | 可选 |

## 本轮：StarRocks 抽样 + GLM-5.3 CodePlan 打标

```bash
export STARROCKS_HOST=127.0.0.1
export STARROCKS_PORT=9030
export STARROCKS_USER=your_user
export STARROCKS_DATABASE=your_database
export STARROCKS_PASSWORD='your_password'

python export_starrocks_candidates.py --output-dir data/candidates/round1
```

然后对 `shard_00.jsonl` … `shard_29.jsonl` 各开一轮 Cursor GLM-5.3 CodePlan：说明书 `prompts/glm_teacher_shard.md`，输出 `data/labeled/round1/shard_XX.jsonl`。

```bash
python validate_labeled_shards.py \
  --candidate-dir data/candidates/round1 \
  --labeled-dir data/labeled/round1 \
  --merged data/labeled/round1_labeled.jsonl \
  --review-dir data/needs_review/round1

python train_student.py \
  --train_file data/labeled/round1_labeled.jsonl \
  --output_dir models/student_v1 \
  --val_ratio 0.1

python compare_teacher_aliyun.py
python evaluate_student_on_gold.py --model_dir models/student_v1
```

## 分步运行

### 1. 冷启动：从存量数据挑选首批候选样本

```bash
python data_prep.py --mode cold_start \
    --input data/raw/bilibili_comments_500w.csv \
    --output data/candidates/round1_candidates.jsonl \
    --sample_size 30000
```

3万条建议按平台/游戏品类分层，覆盖尽量多的语言风格。如果预算允许，
第一轮可以适当多标一些（3-5万条），后续每轮迭代补充几千条即可。

### 2. 双模型交叉标注

```bash
python teacher_labeling.py \
    --input data/candidates/round1_candidates.jsonl \
    --output data/labeled/round1_labeled.jsonl \
    --review_output data/needs_review/round1_review.jsonl
```

跑完后注意看终端打印的"一致率"。如果一致率明显偏低（比如低于70%），
说明这批数据的判断难度较高，建议抽查 review 文件看看分歧点在哪，
必要时调整 `teacher_labeling.py` 里的 SYSTEM_PROMPT 加入更明确的判定规则。

### 3. 微调学生模型（首次训练）

```bash
python train_student.py \
    --train_file data/labeled/round1_labeled.jsonl \
    --output_dir models/student_v1 \
    --val_ratio 0.1
```

训练完成后终端会打印 accuracy、macro_f1，以及每个类别（正面/负面/中性）
各自的 F1，重点关注类别间是否均衡——如果某一类 F1 明显偏低，
大概率是该类别在训练数据里样本量不足，下一轮标注时针对性补充。

打包与分发见上文「把训练好的模型发给别人」。

### 4. 生产环境批量推理（处理每日增量）

```bash
python infer_batch.py \
    --model_dir models/student_v1 \
    --input data/raw/bilibili_daily_20260908.csv \
    --output data/predictions/20260908_predictions.jsonl \
    --review_output data/candidates/20260908_low_confidence.jsonl
```

### 5. 第二轮：LoL 评论主动学习（当前）

从 `data/raw/` 两份英雄联盟赛事微博 CSV，用 `student_v1` 抽 2000 条难例（评论为主，官方博文 ≤10%）：

```bash
python export_round2_candidates.py --model_dir models/student_v1
# 产出 data/candidates/round2/shard_00.jsonl … shard_19.jsonl
```

对每个分片开一轮 Cursor GLM-5.3 CodePlan：说明书 `prompts/glm_teacher_shard_round2.md`，输出 `data/labeled/round2/shard_XX.jsonl`。

```bash
python validate_labeled_shards.py \
  --candidate-dir data/candidates/round2 \
  --labeled-dir data/labeled/round2 \
  --merged data/labeled/round2_labeled.jsonl \
  --review-dir data/needs_review/round2

cat data/labeled/round1_labeled.jsonl data/labeled/round2_labeled.jsonl \
  > data/labeled/round2_train_mixed.jsonl

python train_student.py \
  --train_file data/labeled/round2_train_mixed.jsonl \
  --output_dir models/student_v2 \
  --init_from models/student_v1 \
  --val_ratio 0.1

python evaluate_student_on_gold.py --model_dir models/student_v2
```

中断后可加 `--reuse-cache` 跳过全量打分，直接用 `data/candidates/round2/pool_scored.jsonl` 重新抽样。

## 附加流程：用已有人工标注数据验证蒸馏方案

如果你手头已经有一批人工标注数据（哪怕量不大、哪怕类别不均衡），**不要直接拿来微调**，
先用它验证 `teacher_labeling.py` 里的 prompt 设计靠不靠谱，这个验证成本很低但价值很高。

### 1. 整理成标准格式

```bash
python prepare_validation_gold.py \
    --input "data/raw/你的人工标注文件.xlsx" \
    --output data/validation/validation_gold.jsonl
```

这一步会自动：
- 只保留真正有人工标注的行（很多标注文件里实际有效标注的比例并不高，比如双盲评测文件里
  可能只有20%左右的行有真实人工标签，其余是模型输出）
- 做标签命名归一化（比如"中立/负向/正向" -> 项目统一的"中性/负面/正面"），
  这一步必须显式做，不能假设标注文件用的命名和项目标签体系完全一致
- 打印类别分布、内容类型分布，提示数据缺口（比如某类型内容完全没有标注、
  某个类别样本量过少导致该类别指标不可信）

### 2. 跑双模型交叉标注，和人工标注做对比

```bash
export DASHSCOPE_API_KEY=xxx
export DEEPSEEK_API_KEY=xxx
python evaluate_teacher_labeling.py \
    --input data/validation/validation_gold.jsonl \
    --output data/validation/validation_predictions.jsonl
```

会输出：
- Qwen / DeepSeek 各自单独跟人工标注的一致率、混淆矩阵
- 两模型互相的一致率
- **核心指标**：两模型都判断一致的那部分样本，和人工标注的吻合度——
  因为 `teacher_labeling.py` 实际写入训练集的就是这部分数据，
  如果这个吻合度低于85%左右，说明 prompt 需要先调整，不建议直接大规模标注
- 按内容类型（博文/评论）拆分的准确率，帮你定位问题集中在哪类内容上

### 3. 根据结果决定下一步

- 吻合度达标 -> 可以放心跑 `data_prep.py` 挑选更大规模的候选样本，进入正式标注流程
- 吻合度不达标 -> 回头改 `teacher_labeling.py` 里的 SYSTEM_PROMPT，
  参考混淆矩阵里错得最多的那个类别对（比如"人工标负面，模型判中性"这种最常见的错法），
  针对性补充判定规则，改完重新跑这一步，直到指标达标再上量

## 关键设计取舍说明

1. **为什么标注只输出3分类（正/负/中性），不直接输出细粒度情绪？**
   生产环境需要的是稳定、可解释的分类头。情绪细分类（愤怒/失望/期待等）作为
   `emotion` 字段保留在标注数据里，可以用于后续的舆情分析报表，但暂不作为
   主分类任务的训练目标，避免类别过细导致每类样本量不足、模型学不稳。
   等主链路跑顺后，可以考虑加一个多任务头单独预测情绪细分类。

2. **为什么两个 LLM 交叉验证，而不是用一个模型标完就算？**
   蒸馏效果的上限取决于标签质量，不是模型规模。单模型标注不可避免存在
   系统性偏差（比如对某类反讽表达一直判断错），两模型交叉能过滤掉这类
   系统性错误，同时不一致的样本天然就是"真正有歧义/难度较高"的 case，
   转人工复核性价比最高。

3. **为什么学生模型用 base 尺寸（110M），不用 large？**
   150万-200万条/天的吞吐量要求下，base 尺寸在 GPU 批处理下有明显的
   速度优势，而精度差距在实践中往往没有理论上大——决定效果的核心
   还是训练数据质量。如果小规模 A/B 测试证明 large 版本精度提升明显
   且吞吐量能接受，再考虑升级。

4. **为什么低置信度阈值设为 0.6？**
   这是一个起点值，不是绝对标准。建议上线后观察 confidence 分布，
   如果发现 0.6 挑出来的样本量太大（标注预算扛不住）或太小（漏掉太多
   模型判断错的样本），可以在 `config.py` 里调整
   `LOW_CONFIDENCE_THRESHOLD`。

## 后续可优化方向

- **成本监控**：`teacher_labeling.py` 里可以加一个 token 用量统计，
  方便估算每轮迭代的 API 调用成本
- **人工复核界面**：`needs_review/` 目录里的数据目前需要人工用文本编辑器
  看，量大了之后建议做一个简单的标注管理页面（哪怕是个 Excel 模板）
- **多任务学习**：主链路稳定后，可以在 `train_student.py` 里加一个
  情绪细分类的辅助 loss，用已经收集的 `emotion` 字段做多任务训练
- **模型压缩**：如果吞吐量仍然吃紧，可以考虑对 student 模型做
  量化（INT8）或者进一步蒸馏到更小的模型（如 TinyBERT 结构）

## 开源范围

- 代码、标注说明、训练笔记可以公开。
- 不包含生产评论、金标原文、`.env`、镜像仓库地址和内网连接信息。
- StarRocks 表名（如 `dwd_crawl_weibo_post_details_merged`）是示例仓表，接到自己的库时要改 SQL。
- 仅供个人学习和研究使用。
