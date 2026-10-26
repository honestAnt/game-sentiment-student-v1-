# -*- coding: utf-8 -*-
"""
config.py
全局配置：路径、模型、标签体系、API 参数
所有其他脚本都从这里导入配置，改配置只需要改这一个文件。
"""

import os

# ============ 路径配置 ============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")            # 原始爬取数据存放处
CANDIDATE_DIR = os.path.join(DATA_DIR, "candidates")     # 待标注候选样本（主动学习挑出来的难例）
LABELED_DIR = os.path.join(DATA_DIR, "labeled")          # LLM 标注后的数据
REVIEW_DIR = os.path.join(DATA_DIR, "needs_review")      # 两模型标注不一致 / 低置信度，需人工复核
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "baseline")  # 当前生产环境模型（若无则为 None）
STUDENT_MODEL_OUTPUT = os.path.join(MODEL_DIR, "student_v1")

for d in [DATA_DIR, RAW_DATA_DIR, CANDIDATE_DIR, LABELED_DIR, REVIEW_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============ 标签体系 ============
# 主标签：情感极性，3 分类。这是学生模型真正要学的东西，
# 因为线上推理要求稳定、可解释，细粒度情绪标签作为辅助分析字段，不进分类头。
LABEL2ID = {"负面": 0, "中性": 1, "正面": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

# 辅助情绪标签（仅用于分析和后续扩展，不参与本轮训练）
EMOTION_TAGS = ["愤怒", "失望", "无聊", "期待", "惊喜", "无关/灌水", "其他"]

# ============ 学生模型（生产环境模型）配置 ============
# 优先复用已经在情感任务上适配过的 checkpoint，收敛更快、需要的数据更少
STUDENT_BASE_MODEL = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
MAX_SEQ_LEN = 256          # 游戏弹幕/评论普遍很短，128 足够，太长会拖慢吞吐
TRAIN_BATCH_SIZE = 64      # 有效 batch；CUDA 直接用，MPS 用更小 micro-batch + accum 对齐
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4

# Apple Silicon 16GB 统一内存：GPU/CPU/Cursor 抢同一块内存。micro-batch 必须小，
# 且不要开 gradient checkpointing（MPS 上是带宽杀手，整机会卡死）。
MPS_TRAIN_BATCH_SIZE = 2
MPS_EVAL_BATCH_SIZE = 4
MPS_INFER_BATCH_SIZE = 8
# 0.0 = 不设上限（torch 文档：fraction=0 表示 unlimited）。
# 设 1.0/1.4 时本机「other allocations」会涨到接近上限，backward 仍 OOM；
# 历史 v9 能训完时也未卡死在 watermark。不要与 PYTORCH_MPS_HIGH_WATERMARK_RATIO 环境变量同时设。
MPS_HIGH_WATERMARK_RATIO = 0.0


def mps_training_arg_overrides(effective_train_bs: int = TRAIN_BATCH_SIZE) -> dict:
    """HuggingFace TrainingArguments 的 MPS 覆盖项。不设 use_mps_device（已弃用）。"""
    micro = MPS_TRAIN_BATCH_SIZE
    return {
        "per_device_train_batch_size": micro,
        "per_device_eval_batch_size": MPS_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": max(1, effective_train_bs // micro),
        "dataloader_num_workers": 0,
        "save_total_limit": 2,
    }


def default_infer_batch_size(device: str) -> int:
    if device == "mps":
        return MPS_INFER_BATCH_SIZE
    if device == "cpu":
        return 16
    return 256

# ============ StarRocks（候选导出） ============
COMMENT_SOURCE = "dwd_crawl_weibo_post_comments_details_merged"
POST_SOURCE = "dwd_crawl_weibo_post_details_merged"
ALIYUN_SENTIMENT_TO_LABEL = {1: "正面", 2: "中性", 3: "负面"}

ROUND1_SAMPLE_SIZE = 3000
ROUND1_TARGET_COMMENTS = 1000
ROUND1_TARGET_NEUTRALS = 800
ROUND1_N_SHARDS = 30
ROUND1_SHARD_SIZE = 100
ROUND1_SEED = 42

ROUND2_SAMPLE_SIZE = 2000
ROUND2_N_SHARDS = 20
ROUND2_SHARD_SIZE = 100
ROUND2_SEED = 42
ROUND2_MAX_POST_FRAC = 0.1
ROUND2_HARD_FRAC = 0.7
ROUND2_P_NEG_HARD = 0.20
ROUND2_COMMENT_SOURCE = "weibo_post_comments_details"
ROUND2_POST_SOURCE = "weibo_post_details"

# ============ Round3 主动学习（先试标后放量）============
# round2 教训：70% 难例配额 + 按 p_neg 捞负面 -> 采出负面 44.7%（金标 21%）-> v2 负面先验漂移、
# 真实任务 84%->79%。round3 改为：平台分层随机为主 + 少量低置信难例，打标后做标签分布硬约束。
ROUND3_SAMPLE_SIZE = 300          # 试标 300，人工抽检通过后再放量
ROUND3_N_SHARDS = 3
ROUND3_SHARD_SIZE = 100
ROUND3_SEED = 43
ROUND3_HARD_FRAC = 0.3            # 仅按 conf<LOW_CONFIDENCE_THRESHOLD 判难例，不再按 p_neg 捞负面
ROUND3_NEG_MAX_FRAC = 0.26        # teacher 标完后负面占比超过此值 -> 降采样再混训（金标负面 21-26%）

# round3 放量：配额按 (platform, content_type) 三分。试标教训：weibo 150 条里博文仅 8 条，
# 学生模型只看文本，评论的「直球夸赞→正面」口径泄漏到博文仪式应援（v5 博文 raw -2.7pt）。
# B站 40% 顺带覆盖 raw 抽检发现的直球应援正面召回弱问题。
ROUND3_FULL_SAMPLE_SIZE = 1000
ROUND3_FULL_N_SHARDS = 10
ROUND3_FULL_SHARD_SIZE = 100
ROUND3_FULL_SEED = 44
ROUND3_SLICE_FRAC = {
    ("bilibili", "评论"): 0.4,
    ("weibo", "评论"): 0.3,
    ("weibo", "博文"): 0.3,
}

# ============ Round4 多游戏均衡（v6 失败归因的直接修复）============
# v6 教训：round3_full 全是 LoL 赛事域而金标是多游戏混合（异环/鸣潮等），域内放量不涨金标，
# 且数据变多导致置信度上移（<0.75 从 149 条降到 60 条）冲垮 T0.75 回退安全网。
# round4 数据源改为与金标博文同源同分布的双盲2000未标注行（多游戏微博），修复域偏移；
# 另从 B站存量按 v5 两类已确认弱点定向挖难例（直球应援判中性、黑话反讽判正面）。
ROUND4_SEED = 45
ROUND4_SHARD_SIZE = 100
ROUND4_BILI_HARD_TARGET = 200   # B站定向难例上限（应援/反讽关键词 + agent-v5 分歧，不足随机补）
ROUND4_HOLDOUT_SIZE = 200       # 从多游戏标注中留出做独立阈值验证（不进训练）
ROUND4_HOLDOUT_SEED = 46
ROUND4_BLIND_XLSX = "微博情感双盲对比（2000条）.xlsx"
ROUND4_BILI_1600_XLSX = "LoL评论子agent标注（1600条）_B站.xlsx"
ROUND4_CHEER_PATTERN = r"加油|fighting|lets?\s?go|冲[!！鸭呀啊～~]|夺冠|拿下|必胜|稳住|挺你"
ROUND4_SARCASM_PATTERN = r"捞批捞|定制冠军|定制|缅北|蛆|急孝|典中典|赢麻|闹麻|孝子|就这|垃圾|烂完|摆烂|下头"

# ============ Round5 示例品牌舆情（新域：B站视频+评论 / 小红书笔记+评论）============
# 数据带 相关性(品牌/品类/无关) 与机器预标 sentiment_final；只采相关行。
# 稀缺片（bili评论/bili视频/xhs笔记）全取，xhs评论按机器先验分层随机补足到总量；
# 全池用 student_v5 打分留档 pool_scored（新域置信分布可日后对比，不参与本轮配额——
# 跨域置信作为难例信号不可靠，v5 是游戏域模型）。
ROUND5_SEED = 47
ROUND5_SHARD_SIZE = 100
ROUND5_TOTAL_TARGET = 900          # 候选总量；xhs评论补足余量
ROUND5_TEXT_MIN_LEN = 2
ROUND5_RELEVANCE = ("品牌", "品类")
ROUND5_MCJ_BILI_COMMENTS_CSV = "示例品牌__bili_video_comments_details_sentiment.csv"
ROUND5_MCJ_BILI_VIDEOS_CSV = "示例品牌__bili_video_details_sentiment.csv"
ROUND5_MCJ_XHS_COMMENTS_CSV = "示例品牌__xhs_note_comments_details_sentiment.csv"
ROUND5_MCJ_XHS_NOTES_CSV = "示例品牌__xhs_note_details_sentiment.csv"

# ============ Round6 微博评论短板（gold_v6 分片回退后的蒸馏）============
# 50% weibo评论（约 30% 难例：cheer / raw=正面且 conf<0.75）+ 25% 博文防泄漏 + 25% B站抗议/动员。
# 源：StarRocks DWD 多游戏平表，LoL 不整批排除但单域封顶；金标永不进训练。
ROUND6_SAMPLE_SIZE = 1200
ROUND6_N_SHARDS = 12
ROUND6_SHARD_SIZE = 100
ROUND6_SEED = 48
ROUND6_COMMENT_HARD_FRAC = 0.30
ROUND6_SLICE_FRAC = {
    ("weibo", "评论"): 0.50,
    ("weibo", "博文"): 0.25,
    ("bilibili", "评论"): 0.25,
}
ROUND6_CHEER_PATTERN = (
    ROUND4_CHEER_PATTERN + r"|恭喜|打call|yyds|太强|爱了|名不虚传"
)
ROUND6_PROTEST_PATTERN = (
    r"抵制|联名|道歉|退钱|倒闭|圈钱|别买|卸载|别充|问责|"
    r"出来说话|别洗|骗钱|别氪|退游|别给钱|官方管管"
)

MIN_TRAIN_LABELED = 2000
CLASS_WARN_MIN = 150
REQUEST_ROW_MIN = 15000
REQUEST_ROW_MAX = 25000

STARROCKS_ENV_KEYS = (
    "STARROCKS_HOST",
    "STARROCKS_PORT",
    "STARROCKS_USER",
    "STARROCKS_PASSWORD",
    "STARROCKS_DATABASE",
)


def require_starrocks_env() -> dict:
    missing = [k for k in STARROCKS_ENV_KEYS if not os.environ.get(k)]
    if missing:
        raise SystemExit("缺少环境变量: " + ", ".join(missing))
    return {
        "host": os.environ["STARROCKS_HOST"],
        "port": int(os.environ["STARROCKS_PORT"]),
        "user": os.environ["STARROCKS_USER"],
        "password": os.environ["STARROCKS_PASSWORD"],
        "database": os.environ["STARROCKS_DATABASE"],
        "charset": "utf8mb4",
    }

LOW_CONFIDENCE_THRESHOLD = 0.6   # softmax 最高概率低于这个值，判定为"模型拿不准"

# ============ 教师模型（LLM 标注）配置 ============
# 两个模型交叉标注，只保留一致的结果，不一致的转人工复核
# API Key 从环境变量读取，不要硬编码到代码里
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_NAME = "qwen-plus"          # 标注用途，不要求极致速度，plus 性价比通常够用

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-chat"

LLM_CONCURRENCY = 8         # 并发请求数，根据你的 API 限流配额调整
LLM_MAX_RETRIES = 3

# ============ 平台配置 ============
# 每个平台的语言风格不同，标注 prompt 里会注入这段说明，
# 让 LLM 判断时对该平台的表达习惯有认知
PLATFORM_CONTEXT = {
    "bilibili": "内容来自B站游戏区的弹幕或评论，文本通常很短，包含大量缩写、玩梗、游戏黑话（如'典中典'、'急急急'、'下头'），语气可能阴阳怪气或反讽。",
    "weibo": "内容来自微博关于游戏的博文或评论，文本长度中等，可能包含话题标签(#)和@提及，情绪化表达较多。",
    "xiaohongshu": "内容来自小红书关于游戏的笔记或评论，偏种草/测评风格，情感表达相对完整、口语化。",
    "douyin": "内容来自抖音游戏相关视频的评论，文本很短，情绪化程度高，也存在大量与视频内容无关的灌水评论（如'求关注''第一个赞'）。",
}
