from typing import Dict, List
from src.state_of_mind.types.perception import ValidationRule
from src.state_of_mind.utils.data_validator import IS_DICT, IS_STR, IS_LIST, IS_BOOL

OTHER = "other"

# 模板类别
CATEGORY_RAW = "raw"
CATEGORY_SUGGESTION = "suggestion"
COREFERENCE_RESOLUTION_BATCH = "coreference_resolution_batch"
GLOBAL_SEMANTIC_SIGNATURE = "global_semantic_signature"

# prompt 类型
# 预处理 - 并行
PARALLEL_PREPROCESSING = "parallel_preprocessing"
# 感知 - 并行
PARALLEL_PERCEPTION = "parallel_perception"
# 高阶 - 并行
PARALLEL_HIGH_ORDER = "parallel_high_order"
# 建议 - 串行
SERIAL_SUGGESTION = "serial_suggestion"

# --- 常量存储 ---
# === 按类型分组 ===
PARALLEL_PREPROCESSING_STEPS: Dict[str, dict] = {}
PARALLEL_PERCEPTION_STEPS: Dict[str, dict] = {}
PARALLEL_HIGH_ORDER_STEPS: Dict[str, dict] = {}
SERIAL_SUGGESTION_STEPS: Dict[str, dict] = {}
# === 辅助集合 ===
PERCEPTION_LAYERS: set[str] = set()
PARALLEL_PREPROCESSING_KEYS: set[str] = set()
PARALLEL_PERCEPTION_KEYS: set[str] = set()
PARALLEL_HIGH_ORDER_KEYS: set[str] = set()
SERIAL_SUGGESTION_KEYS: set[str] = set()
# === 前端全量步骤配置 ===
ALL_STEPS_FOR_FRONTEND: List[dict] = []
# --- 初始化标志（模块级，天然全局）---
PIPELINE_INITIALIZED = False

# 大模型预处理
LLM_PARTICIPANTS_EXTRACTION = "LLM_PARTICIPANTS_EXTRACTION"
LLM_DIMENSION_GATE = "LLM_DIMENSION_GATE"
LLM_INFERENCE_ELIGIBILITY = "LLM_INFERENCE_ELIGIBILITY"

# 大模型感知层
LLM_PERCEPTION_TEMPORAL_EXTRACTION = "LLM_PERCEPTION_TEMPORAL_EXTRACTION"
LLM_PERCEPTION_SPATIAL_EXTRACTION = "LLM_PERCEPTION_SPATIAL_EXTRACTION"
LLM_PERCEPTION_VISUAL_EXTRACTION = "LLM_PERCEPTION_VISUAL_EXTRACTION"
LLM_PERCEPTION_AUDITORY_EXTRACTION = "LLM_PERCEPTION_AUDITORY_EXTRACTION"
LLM_PERCEPTION_OLFACTORY_EXTRACTION = "LLM_PERCEPTION_OLFACTORY_EXTRACTION"
LLM_PERCEPTION_TACTILE_EXTRACTION = "LLM_PERCEPTION_TACTILE_EXTRACTION"
LLM_PERCEPTION_GUSTATORY_EXTRACTION = "LLM_PERCEPTION_GUSTATORY_EXTRACTION"
LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION = "LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION"
LLM_PERCEPTION_COGNITIVE_EXTRACTION = "LLM_PERCEPTION_COGNITIVE_EXTRACTION"
LLM_PERCEPTION_BODILY_EXTRACTION = "LLM_PERCEPTION_BODILY_EXTRACTION"
LLM_PERCEPTION_EMOTIONAL_EXTRACTION = "LLM_PERCEPTION_EMOTIONAL_EXTRACTION"
LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION = "LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION"

# 大模型推理层
LLM_STRATEGY_ANCHOR = "LLM_STRATEGY_ANCHOR"
LLM_CONTRADICTION_MAP = "LLM_CONTRADICTION_MAP"
LLM_MANIPULATION_DECODE = "LLM_MANIPULATION_DECODE"
LLM_MINIMAL_VIABLE_ADVICE = "LLM_MINIMAL_VIABLE_ADVICE"

# 定义各阶段并行感知任务允许使用的上下文 marker
ALLOWED_PARALLEL_PERCEPTION_MARKERS = {
    0: {"### USER_INPUT BEGIN"},
    1: {"### USER_INPUT BEGIN"},
    2: {"### USER_INPUT BEGIN"},
    3: {"### USER_INPUT BEGIN"},
    4: {"### USER_INPUT BEGIN"},
    5: {"### USER_INPUT BEGIN"},
    6: {"### USER_INPUT BEGIN"},
    7: {"### USER_INPUT BEGIN"},
    8: {"### USER_INPUT BEGIN"},
    9: {"### USER_INPUT BEGIN"},
    10: {"### USER_INPUT BEGIN"},
    11: {"### USER_INPUT BEGIN"}
}

# 定义各阶段并行高阶任务允许使用的上下文 marker
ALLOWED_PARALLEL_HIGH_ORDER_MARKERS = {
    0: {  # 策略锚定
        "### PERCEPTUAL_CONTEXT_BATCH BEGIN",
        "### LEGITIMATE_PARTICIPANTS BEGIN"
    },
    1: {  # 矛盾暴露
        "### PERCEPTUAL_CONTEXT_BATCH BEGIN",
        "### LEGITIMATE_PARTICIPANTS BEGIN"
    },
    2: {  # 操控机制解码
        "### PERCEPTUAL_CONTEXT_BATCH BEGIN",
        "### LEGITIMATE_PARTICIPANTS BEGIN"
    }
}

# 定义各阶段串行任务允许使用的上下文 marker
ALLOWED_SERIAL_SUGGESTION_MARKERS = {
    0: {  # 最小可行性建议
        "### STRATEGY_ANCHOR_CONTEXT BEGIN",
        "### CONTRADICTION_MAP_CONTEXT BEGIN",
        "### MANIPULATION_DECODE_CONTEXT BEGIN",
        "### LEGITIMATE_PARTICIPANTS BEGIN"
    },
}

# 默认 API URL 映射
DEFAULT_API_URLS = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
}

# 语义上等同于“无信息”的字符串，视为应清除的占位符
SEMANTIC_NULL_STRINGS = frozenset([
    "未提及", "未知", "待定", "不清楚", "无", "没有", "暂无", "不详", "未说明", "无明确描述", "无描述", "没有描述", "没有描述内容",
    "none", "unknown", "unspecified", "n/a", "na", "—", "-", "…", "..."
])

# 合法代词，暂时不再使用
CHINESE_PRONOUNS = {
    # 第一人称单数
    "我", "吾", "余", "予", "俺", "咱", "本人", "自己", "自身", "个人",
    "鄙人", "小人", "不才", "在下", "晚生", "晚辈", "老朽", "老夫", "老汉",
    "本官", "本座", "本王", "本宫", "本尊", "本帅", "本将",

    # 第二人称单数
    "你", "您", "汝", "尔", "君", "卿", "阁下", "足下", "台端", "尊驾",
    "贵方", "贵客", "贵宾", "这位", "那位",

    # 第三人称单数
    "他", "她", "它", "牠", "祂", "彼", "其", "该人", "此人", "其人", "该者",
    "该员", "该方", "该个体", "该主体", "所述人", "前述人", "该对象",
    "这位", "那位", "这人", "那人", "这个人", "那个人", "此君", "该君",

    # 第一人称复数
    "我们", "咱们", "我等", "我辈", "我曹", "吾等", "吾辈", "吾曹",
    "本队", "本方", "本组", "本团", "本派", "本门", "本帮", "本教",

    # 第二人称复数
    "你们", "您们", "汝等", "尔等", "诸位", "各位", "列位", "众位",
    "大家", "大伙", "大伙儿", "大家伙",

    # 第三人称复数
    "他们", "她们", "它们", "牠们", "祂们", "彼等", "其等", "该等",
    "此辈", "该批", "该群", "该组", "该队", "该伙", "该帮", "该派",

    # 泛指但可能指向具体参与者
    "双方", "各方", "彼此", "对方", "对手", "敌手", "同伴", "同伴们",
    "同伙", "同伙们", "队友", "队友们", "同伴", "同伴们",

    # 口语/方言变体
    "俺们", "咱家", "阿拉", "侬", "伊", "渠", "怹",

    # 网络用语
    "偶", "额", "窝", "伦家", "本宝宝", "朕", "寡人"
}

# 明确应排除的代词（即使出现在事件中也不映射）
EXCLUDED_PRONOUNS = {
    # 泛指人群
    "别人", "他人", "其他人", "旁人", "外人", "某人", "某些人", "有人", "有些人",
    "任何人", "每个人", "所有人", "众人", "大众", "群众", "人群", "人们", "人类",
    "世人", "天下人", "百姓", "民众", "人民", "公众", "老百姓",

    # 抽象指代
    "谁", "何人", "什么人", "哪位", "何者", "孰",
    "这个", "那个", "这些", "那些", "此", "彼",

    # 不确定指代
    "有的人", "部分人", "多数人", "少数人", "许多人", "不少人", "大多数人",
    "绝大多数人", "几乎所有", "每一个", "各自", "各自的人",

    # 职业/角色泛指
    "警察", "医生", "老师", "学生", "工人", "农民", "商人", "官员",
    "军人", "记者", "律师", "演员", "作家", "艺术家", "科学家",

    # 关系泛指
    "朋友", "敌人", "亲人", "家人", "亲戚", "邻居", "同事", "同学", "战友",
    "同伴", "伙伴", "对手", "竞争者", "合作者",

    # 群体类别
    "男人", "女人", "男孩", "女孩", "儿童", "少年", "青年", "中年", "老年",
    "老人", "年轻人", "成年人", "未成年人", "男性", "女性",

    # 社会身份泛指
    "领导", "上司", "下属", "员工", "职员", "成员", "参与者", "观众", "听众",
    "读者", "用户", "客户", "顾客", "消费者", "患者", "病人",

    # 方位指代
    "这里的人", "那里的人", "这边的人", "那边的人", "当地的人", "现场的人",
    "周围的人", "附近的人", "身边的人",

    # 时间指代
    "当时的人", "那时的人", "现在的人", "过去的人", "未来的人",

    # 完全模糊
    "某个", "某些", "某种", "某类", "某位", "某方", "某群体", "某组织"
}

REQUIRED_FIELDS_BY_CATEGORY: Dict[str, Dict[str, List[ValidationRule]]] = {
    CATEGORY_RAW: {
        # ────────────────────────────────────────
        # 1. 源数据提取（参与者列表）—— 你已确认，保留不变
        # ────────────────────────────────────────
        LLM_PARTICIPANTS_EXTRACTION: [
            # === 核心标识 ===
            ("participants", True, IS_LIST, "participants（参与者列表）："),
            ("participants.*.entity", True, IS_STR, "entity（原始指称短语）："),

            # === 静态社会属性 ===
            ("participants.*.social_role", False, IS_STR, "social_role（非职业的社会身份标签）："),
            ("participants.*.occupation", False, IS_STR, "occupation（职业身份）："),
            ("participants.*.family_status", False, IS_STR, "family_status （家庭状态标签）："),
            ("participants.*.education_level", False, IS_STR, "education_level（教育程度）："),
            ("participants.*.cultural_identity", False, IS_STR, "cultural_identity（文化/民族身份标签）："),
            ("participants.*.primary_language", False, IS_STR, "primary_language（主要使用语言）："),
            ("participants.*.institutional_affiliation", False, IS_STR, "institutional_affiliation（所属机构标签）："),

            # === 生物与生理属性 ===
            ("participants.*.age_range", False, IS_STR, "age_range（年龄范围）："),
            ("participants.*.gender", False, IS_STR, "gender（性别或相关表述）："),
            ("participants.*.ethnicity_or_origin", False, IS_STR, "ethnicity_or_origin（族群/国籍/地域出身）："),
            ("participants.*.physical_traits", False, IS_LIST, "physical_traits（固有生理特征）："),
            ("participants.*.current_action_state", False, IS_STR, "current_action_state（当前状态）："),
            ("participants.*.visible_injury_or_wound", False, IS_STR, "visible_injury_or_wound（可见伤痕或包扎）："),

            # === 感官与外显特征 ===
            ("participants.*.appearance", False, IS_LIST, "appearance（外貌与衣着特征）："),
            ("participants.*.voice_quality", False, IS_STR, "voice_quality（嗓音特质）："),
            ("participants.*.inherent_odor", False, IS_STR, "inherent_odor（体味或气息）："),
            ("participants.*.possessions", False, IS_LIST, "possessions（物理持有或佩戴物品）："),
            ("participants.*.speech_pattern", False, IS_STR, "speech_pattern（说话方式）："),

            # === 交互与场景角色 ===
            ("participants.*.interaction_role", False, IS_STR, "interaction_role（当前互动中的临时角色）："),
        ],

        # ────────────────────────────────────────
        # 2. 动态筛选存在哪些感知维度信息
        # ────────────────────────────────────────
        LLM_DIMENSION_GATE: [
            ("pre_screening", True, IS_DICT, "pre_screening（预筛选）："),
            ("pre_screening.temporal", True, IS_BOOL, "temporal（时间感知）："),
            ("pre_screening.spatial", True, IS_BOOL, "spatial（空间感知）："),
            ("pre_screening.visual", True, IS_BOOL, "visual（视觉感知）："),
            ("pre_screening.auditory", True, IS_BOOL, "auditory（听觉感知）："),
            ("pre_screening.olfactory", True, IS_BOOL, "olfactory（嗅觉感知）："),
            ("pre_screening.tactile", True, IS_BOOL, "tactile（触觉感知）："),
            ("pre_screening.gustatory", True, IS_BOOL, "gustatory（味觉感知）："),
            ("pre_screening.interoceptive", True, IS_BOOL, "interoceptive（内感受感知）："),
            ("pre_screening.cognitive", True, IS_BOOL, "cognitive（认知过程感知）："),
            ("pre_screening.bodily", True, IS_BOOL, "bodily（躯体化表现感知）："),
            ("pre_screening.emotional", True, IS_BOOL, "emotional（情感状态感知）："),
            ("pre_screening.social_relation", True, IS_BOOL, "social_relation（社会关系感知）："),
        ],

        # ────────────────────────────────────────
        # 3. 提前判断是否值得开启高阶四步链
        # ────────────────────────────────────────
        LLM_INFERENCE_ELIGIBILITY: [
            ("eligibility", True, IS_DICT, "eligibility（高阶步骤执行资格）："),
            ("eligibility.eligible", True, IS_BOOL, "eligible（符合条件）："),
        ],

        # ────────────────────────────────────────
        # 4. 时间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TEMPORAL_EXTRACTION: [
            ("temporal", False, IS_DICT, "temporal（时间信息根对象）："),
            ("temporal.events", False, IS_LIST, "events（时间事件列表）："),
            ("temporal.summary", False, IS_STR, "summary（时间事件摘要）："),

            ("temporal.events.*.experiencer", False, IS_STR, "experiencer（事件陈述主体）："),
            ("temporal.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("temporal.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化时间语义标识）："),
            ("temporal.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("temporal.events.*.temporal_mentions", False, IS_LIST, "temporal_mentions（显示时间成分）："),
            ("temporal.events.*.temporal_mentions.*.phrase", False, IS_STR, "phrase（原文时间短语）："),
            ("temporal.events.*.temporal_mentions.*.type", False, IS_STR, "type（时间成分类型）："),
        ],

        # ────────────────────────────────────────
        # 5. 空间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_SPATIAL_EXTRACTION: [
            ("spatial", False, IS_DICT, "spatial（空间信息根对象）："),
            ("spatial.events", False, IS_LIST, "events（空间事件列表）："),
            ("spatial.summary", False, IS_STR, "summary（空间事件摘要）："),

            ("spatial.events.*.experiencer", False, IS_STR, "experiencer（空间陈述主体）："),
            ("spatial.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("spatial.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化空间语义标识）："),
            ("spatial.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("spatial.events.*.spatial_mentions", False, IS_LIST, "spatial_mentions（显式空间成分）："),
            ("spatial.events.*.spatial_mentions.*.phrase", False, IS_STR, "phrase（原文空间短语）："),
            ("spatial.events.*.spatial_mentions.*.type", False, IS_STR, "type（空间成分类型）："),
        ],

        # ────────────────────────────────────────
        # 6. 视觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_VISUAL_EXTRACTION: [
            ("visual", False, IS_DICT, "visual（视觉信息根对象）："),
            ("visual.events", False, IS_LIST, "events（视觉事件列表）："),
            ("visual.summary", False, IS_STR, "summary（视觉事件摘要）："),

            ("visual.events.*.experiencer", False, IS_STR, "experiencer（视觉陈述主体）："),
            ("visual.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("visual.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化视觉语义标识）："),
            ("visual.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("visual.events.*.visual_mentions", False, IS_LIST, "visual_mentions（显式视觉成分）："),
            ("visual.events.*.visual_mentions.*.phrase", False, IS_STR, "phrase（原文视觉短语）："),
            ("visual.events.*.visual_mentions.*.type", False, IS_STR, "type（视觉成分类型）：")
        ],

        # ────────────────────────────────────────
        # 7. 听觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_AUDITORY_EXTRACTION: [
            ("auditory", False, IS_DICT, "auditory（听觉信息根对象）："),
            ("auditory.events", False, IS_LIST, "events（听觉事件列表）："),
            ("auditory.summary", False, IS_STR, "summary（听觉事件摘要）："),

            ("auditory.events.*.experiencer", False, IS_STR, "experiencer（听觉陈述主体）："),
            ("auditory.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("auditory.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化听觉语义标识）："),
            ("auditory.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("auditory.events.*.auditory_mentions", False, IS_LIST, "auditory_mentions（显式听觉成分）："),
            ("auditory.events.*.auditory_mentions.*.phrase", False, IS_STR, "phrase（原文听觉短语）："),
            ("auditory.events.*.auditory_mentions.*.type", False, IS_STR, "type（听觉成分类型）：")
        ],

        # ────────────────────────────────────────
        # 8. 嗅觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_OLFACTORY_EXTRACTION: [
            ("olfactory", False, IS_DICT, "olfactory（嗅觉信息根对象）："),
            ("olfactory.events", False, IS_LIST, "events（嗅觉事件列表）："),
            ("olfactory.summary", False, IS_STR, "summary（嗅觉事件摘要）："),

            ("olfactory.events.*.experiencer", False, IS_STR, "experiencer（嗅觉陈述主体）："),
            ("olfactory.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("olfactory.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化嗅觉语义标识）："),
            ("olfactory.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("olfactory.events.*.olfactory_mentions", False, IS_LIST, "olfactory_mentions（显式嗅觉成分）："),
            ("olfactory.events.*.olfactory_mentions.*.phrase", False, IS_STR, "phrase（原文嗅觉短语）："),
            ("olfactory.events.*.olfactory_mentions.*.type", False, IS_STR, "type（嗅觉成分类型）："),
        ],

        # ────────────────────────────────────────
        # 9. 触觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TACTILE_EXTRACTION: [
            ("tactile", False, IS_DICT, "tactile（触觉信息根对象）："),
            ("tactile.events", False, IS_LIST, "events（触觉事件列表）："),
            ("tactile.summary", False, IS_STR, "summary（触觉事件摘要）："),

            ("tactile.events.*.experiencer", False, IS_STR, "experiencer（触觉陈述主体）："),
            ("tactile.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("tactile.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化触觉语义标识）："),
            ("tactile.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("tactile.events.*.tactile_mentions", False, IS_LIST, "tactile_mentions（显式触觉成分）："),
            ("tactile.events.*.tactile_mentions.*.phrase", False, IS_STR, "phrase（原文触觉短语）："),
            ("tactile.events.*.tactile_mentions.*.type", False, IS_STR, "type（触觉成分类型）："),
        ],

        # ────────────────────────────────────────
        # 10. 味觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_GUSTATORY_EXTRACTION: [
            ("gustatory", False, IS_DICT, "gustatory（味觉信息根对象）："),
            ("gustatory.events", False, IS_LIST, "events（味觉事件列表）："),
            ("gustatory.summary", False, IS_STR, "summary（味觉事件摘要）："),

            ("gustatory.events.*.experiencer", False, IS_STR, "experiencer（味觉陈述主体）："),
            ("gustatory.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("gustatory.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化味觉语义标识）："),
            ("gustatory.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("gustatory.events.*.gustatory_mentions", False, IS_LIST, "gustatory_mentions（显式味觉成分）："),
            ("gustatory.events.*.gustatory_mentions.*.phrase", False, IS_STR, "phrase（原文味觉短语）："),
            ("gustatory.events.*.gustatory_mentions.*.type", False, IS_STR, "type（味觉成分类型）："),
        ],

        # ────────────────────────────────────────
        # 11. 内感受
        # ────────────────────────────────────────
        LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION: [
            ("interoceptive", False, IS_DICT, "interoceptive（内感受信息根对象）："),
            ("interoceptive.events", False, IS_LIST, "events（内感受事件列表）："),
            ("interoceptive.summary", False, IS_STR, "summary（内感受事件摘要）："),

            ("interoceptive.events.*.experiencer", False, IS_STR, "experiencer（内感受陈述主体）："),
            ("interoceptive.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("interoceptive.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化内感受语义标识）："),
            ("interoceptive.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("interoceptive.events.*.interoceptive_mentions", False, IS_LIST, "interoceptive_mentions（显式内感受成分）："),
            ("interoceptive.events.*.interoceptive_mentions.*.phrase", False, IS_STR, "phrase（原文内感受短语）："),
            ("interoceptive.events.*.interoceptive_mentions.*.type", False, IS_STR, "type（内感受成分类型）："),
        ],

        # ────────────────────────────────────────
        # 12. 认知过程
        # ────────────────────────────────────────
        LLM_PERCEPTION_COGNITIVE_EXTRACTION: [
            ("cognitive", False, IS_DICT, "cognitive（认知过程信息根对象）："),
            ("cognitive.events", False, IS_LIST, "events（认知事件列表）："),
            ("cognitive.summary", False, IS_STR, "summary（认知过程事件摘要）："),

            ("cognitive.events.*.experiencer", False, IS_STR, "experiencer（认知过程陈述主体）："),
            ("cognitive.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("cognitive.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化认知过程语义标识）："),
            ("cognitive.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("cognitive.events.*.cognitive_mentions", False, IS_LIST, "cognitive_mentions（显式认知过程成分）："),
            ("cognitive.events.*.cognitive_mentions.*.phrase", False, IS_STR, "phrase（原文认知过程短语）："),
            ("cognitive.events.*.cognitive_mentions.*.type", False, IS_STR, "type（认知过程成分类型）："),
        ],

        # ────────────────────────────────────────
        # 13. 躯体化表现
        # ────────────────────────────────────────
        LLM_PERCEPTION_BODILY_EXTRACTION: [
            ("bodily", False, IS_DICT, "bodily（躯体化表现信息根对象）："),
            ("bodily.events", False, IS_LIST, "events（躯体化事件列表）："),
            ("bodily.summary", False, IS_STR, "summary（躯体化表现事件摘要）："),

            ("bodily.events.*.experiencer", False, IS_STR, "experiencer（躯体化行为执行主体）："),
            ("bodily.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("bodily.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化躯体化表现语义标识）："),
            ("bodily.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("bodily.events.*.bodily_mentions", False, IS_LIST, "bodily_mentions（显式躯体化表现成分）："),
            ("bodily.events.*.bodily_mentions.*.phrase", False, IS_STR, "phrase（原文躯体化表现短语）："),
            ("bodily.events.*.bodily_mentions.*.type", False, IS_STR, "type（躯体化表现成分类型）："),
        ],

        # ────────────────────────────────────────
        # 14. 情感状态
        # ────────────────────────────────────────
        LLM_PERCEPTION_EMOTIONAL_EXTRACTION: [
            ("emotional", False, IS_DICT, "emotional（情感状态信息根对象）："),
            ("emotional.events", False, IS_LIST, "events（情感事件列表）："),
            ("emotional.summary", False, IS_STR, "summary（情感状态事件摘要）："),

            ("emotional.events.*.experiencer", False, IS_STR, "experiencer（情绪经历主体）："),
            ("emotional.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("emotional.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化情感状态语义标识）："),
            ("emotional.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            ("emotional.events.*.emotional_mentions", False, IS_LIST, "emotional_mentions（显式情感状态成分）："),
            ("emotional.events.*.emotional_mentions.*.phrase", False, IS_STR, "phrase（原文情感状态短语）："),
            ("emotional.events.*.emotional_mentions.*.type", False, IS_STR, "type（情感状态成分类型）："),
        ],

        # ────────────────────────────────────────
        # 15. 社会关系
        # ────────────────────────────────────────
        LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION: [
            ("social_relation", False, IS_DICT, "social_relation（社会关系信息根对象）："),
            ("social_relation.events", False, IS_LIST, "events（社会关系事件列表）："),
            ("social_relation.summary", False, IS_STR, "summary（社会关系事件摘要）："),

            ("social_relation.events.*.experiencer", False, IS_STR, "experiencer（社会关系经历主体）："),
            ("social_relation.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化社会关系语义标识）："),
            ("social_relation.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("social_relation.events.*.event_markers", False, IS_LIST, "event_markers（核心词汇）："),

            (
                "social_relation.events.*.social_relation_mentions", False, IS_LIST,
                "social_relation_mentions（显式社会关系成分）："),
            ("social_relation.events.*.social_relation_mentions.*.phrase", False, IS_STR, "phrase（原文社会关系短语）："),
            ("social_relation.events.*.social_relation_mentions.*.type", False, IS_STR, "type（社会关系成分类型）："),
        ],

        # ────────────────────────────────────────
        # 16. 策略锚定
        # ────────────────────────────────────────
        LLM_STRATEGY_ANCHOR: [
            ("strategy_anchor", False, IS_DICT, "strategy_anchor（策略锚定根对象）："),
            ("strategy_anchor.events", False, IS_LIST, "events（策略锚定事件列表）："),
            ("strategy_anchor.events.*.agent", False, IS_STR, "agent（行为主体）："),
            ("strategy_anchor.events.*.target", False, IS_STR, "target（作用对象）："),
            ("strategy_anchor.events.*.explicit_justification", False, IS_STR, "explicit_justification（表面理由或公开声明）："),
            ("strategy_anchor.events.*.implicit_goal", False, IS_STR, "implicit_goal（可证伪的隐性目标）："),
            ("strategy_anchor.events.*.behavior", False, IS_STR, "behavior（关键策略性行为）："),
            ("strategy_anchor.events.*.social_script", False, IS_STR, "social_script（所利用的社会/道德脚本）："),
            ("strategy_anchor.events.*.power_differential", False, IS_STR, "power_differential（权力/地位差异基础）："),
            ("strategy_anchor.events.*.audience_role", False, IS_STR, "audience_role（第三方观众在策略中的功能）："),
            ("strategy_anchor.events.*.anchor_perceptions", False, IS_LIST,
             "anchor_perceptions（引用的底层感知事件 semantic_notation 列表）："),
            ("strategy_anchor.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("strategy_anchor.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化策略锚定语义标识）："),

            ("strategy_anchor.synthesis", False, IS_STR, "synthesis（策略层面的全局研判）："),
        ],

        # ────────────────────────────────────────
        # 17. 矛盾暴露
        # ────────────────────────────────────────
        LLM_CONTRADICTION_MAP: [
            ("contradiction_map", False, IS_DICT, "contradiction_map（矛盾暴露根对象）："),
            ("contradiction_map.events", False, IS_LIST, "events（矛盾暴露事件列表）："),
            ("contradiction_map.events.*.claimed_premise", False, IS_STR, "claimed_premise（声称的前提、动机或价值观）："),
            ("contradiction_map.events.*.actual_behavior", False, IS_STR, "actual_behavior（实际采取的行为或安排）："),
            ("contradiction_map.events.*.contradiction_type", False, IS_STR,
             "contradiction_type（矛盾类型：means_end_mismatch / speech_action_split / context_violation / value_inconsistency）："),
            ("contradiction_map.events.*.logical_conflict", False, IS_STR, "logical_conflict（逻辑冲突的精炼陈述）："),
            ("contradiction_map.events.*.anchor_perceptions", False, IS_LIST,
             "anchor_perceptions（引用的底层感知事件 semantic_notation 列表）："),
            ("contradiction_map.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("contradiction_map.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化矛盾暴露语义标识）："),

            ("contradiction_map.synthesis", False, IS_STR, "synthesis（矛盾层面的全局研判）："),
        ],

        # ────────────────────────────────────────
        # 18. 操控机制解码
        # ────────────────────────────────────────
        LLM_MANIPULATION_DECODE: [
            ("manipulation_decode", False, IS_DICT, "manipulation_decode（操控机制解码根对象）："),
            ("manipulation_decode.events", False, IS_LIST, "events（操控机制事件列表）："),
            ("manipulation_decode.events.*.mechanism_type", False, IS_STR,
             "mechanism_type（操控机制大类：guilt_induction / fear_appeal / love_bombing / choice_elimination / reputation_leverage / false_generosity / public_shaming_avoidance）："),
            ("manipulation_decode.events.*.technique", False, IS_STR, "technique（具体实施技术）："),
            ("manipulation_decode.events.*.leverage_point", False, IS_STR, "leverage_point（所利用的心理/社会杠杆点）："),
            ("manipulation_decode.events.*.intended_effect", False, IS_STR, "intended_effect（预期达成的心理或社会效果）："),
            ("manipulation_decode.events.*.exit_barrier_created", False, IS_STR,
             "exit_barrier_created（所制造的退出障碍）："),
            ("manipulation_decode.events.*.anchor_perceptions", False, IS_LIST,
             "anchor_perceptions（引用的底层感知事件 semantic_notation 列表）："),
            ("manipulation_decode.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("manipulation_decode.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化操控机制解码语义标识）："),

            ("manipulation_decode.synthesis", False, IS_STR, "synthesis（操控机制的全局研判）："),
        ],

        # ────────────────────────────────────────
        # 19. 最小可行性建议
        # ────────────────────────────────────────
        LLM_MINIMAL_VIABLE_ADVICE: [
            ("minimal_viable_advice", False, IS_DICT, "minimal_viable_advice（最小可行性建议根对象）："),
            ("minimal_viable_advice.events", False, IS_LIST, "events（建议事件列表）："),
            ("minimal_viable_advice.events.*.counter_action", False, IS_STR, "counter_action（可执行的破局行动）："),
            ("minimal_viable_advice.events.*.targeted_mechanism", False, IS_STR, "targeted_mechanism（所针对的操控机制或环节）："),
            ("minimal_viable_advice.events.*.expected_disruption", False, IS_STR, "expected_disruption（预期破坏的机制效果）："),
            ("minimal_viable_advice.events.*.feasibility_condition", False, IS_STR, "feasibility_condition（行动可行的前提条件）："),
            ("minimal_viable_advice.events.*.anchor_perceptions", False, IS_LIST, "anchor_perceptions（引用的底层感知事件 semantic_notation 列表）："),
            ("minimal_viable_advice.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("minimal_viable_advice.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化最小可行性建议语义标识）："),

            ("minimal_viable_advice.synthesis", False, IS_STR, "synthesis（建议层面的全局总结）："),
        ]
    }
}

# === 基线策略===
STRICT_IRON_LAW_POLICY = {
    "enforce_field_existence": True,
    "enforce_literal_extraction": True,
    "enforce_clean_json": True,
}

# === 步骤级策略覆盖表（按 step 名称定制）===
STEP_POLICY_OVERRIDES: Dict[str, Dict] = {
    # —————— 预处理层——————
    "LLM_DIMENSION_GATE": {
        "enforce_field_existence": False,
        "enforce_literal_extraction": False
    },
    "LLM_INFERENCE_ELIGIBILITY": {
        "enforce_field_existence": False,
        "enforce_literal_extraction": False
    },

    "LLM_PERCEPTION_TEMPORAL_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_SPATIAL_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_VISUAL_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_AUDITORY_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_OLFACTORY_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_TACTILE_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_GUSTATORY_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_COGNITIVE_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_BODILY_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_EMOTIONAL_EXTRACTION": {"enforce_literal_extraction": False},
    "LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION": {"enforce_literal_extraction": False},

}


def get_effective_policy(step_name: str) -> Dict:
    """合并基线策略与步骤特例"""
    base = STRICT_IRON_LAW_POLICY.copy()
    override = STEP_POLICY_OVERRIDES.get(step_name, {})
    base.update(override)
    return base


def render_iron_law_from_policy(policy: Dict) -> str:
    """将策略字典渲染为自然语言铁律文本"""
    lines = []
    if policy.get("enforce_field_existence", True):
        lines.append(
            "**绝对存在性**：判定一个字段是否输出的唯一标准，是看其值是否拥有‘合法的原文依据’。‘合法依据’包括：1) 原文直接陈述的字面内容；2) 基于原文进行直接或间接的、逻辑必然的合理推演结果。若某字段在原文及其逻辑推演范围内均无对应依据，则该字段（含键名）必须彻底省略，严禁填充占位符（如'未知'、null、空字符串）。"
        )

    if policy.get("enforce_literal_extraction", True):
        lines.append(
            "**精准推演边界**：所有推演必须严格基于当前语境内已呈现的信息（包括显性陈述、行为描述、指称关系、情绪反应及重复模式），通过逻辑必要性或唯一合理解释导出结论。允许进行代词消解、意图反推、心理机制建模、社会脚本调用、行为模式归纳等高阶推理，但禁止依赖任何未在语境中支持的外部知识、常识假设、统计规律或价值预设。若移除所依赖的语境证据，推论即不成立，则该推论合法；否则视为过度脑补。")

    if policy.get("enforce_clean_json", True):
        lines.append("**结构纯净**：输出必须是一个紧凑的、合法的 JSON 对象，且仅包含此 JSON，无任何额外文本、说明或标记。")

    return "\n".join(lines)


MENTION_TYPES_CONFIG = {
    "temporal": {
        "frequency", "range", "relative", "cultural", "duration",
        "absolute", "negated", "uncertain"
    },
    "spatial": {
        "relative", "direction", "topological", "toward",
        "cultural", "negated", "measure", "location",
        "layout"
    },
    "visual": {
        "color", "brightness", "expression", "posture",
        "object", "entity", "optical", "negated", "gaze",
        "contact", "medium", "occlusion", "lighting"
    },
    "auditory": {
        "verb", "speech", "sound", "intensity", "prosody",
        "medium", "background", "source", "negated"
    },
    "olfactory": {
        "odor", "source", "intensity", "valence",
        "negated", "action", "category"
    },
    "tactile": {
        "pain", "target", "body", "descriptor", "intensity",
        "texture", "temperature", "motion",
        "vibration", "moisture", "mode", "negated"
    },
    "gustatory": {
        "source", "basic", "complex", "thermal", "evaluation", "intensity",
        "body", "negated"
    },
    "interoceptive": {
        "body", "cardiac", "respiratory", "gastrointestinal", "thermal",
        "muscular", "visceral", "dizziness", "nausea", "fatigue",
        "thirst", "intensity", "initiator", "negated"
    },
    "cognitive": {
        "belief", "intention", "inference", "memory", "doubt",
        "evaluation", "solving", "meta", "certainty", "negated"
    },
    "bodily": {
        "movement", "posture", "facial", "vocal", "autonomic",
        "freeze", "faint", "action", "intensity", "negated"
    },
    "emotional": {
        "emotion", "valence", "arousal", "intensity", "mode",
        "verb", "adjective", "adverb", "mixed"
    },
    "social_relation": {
        "kinship", "role", "address", "possessive", "relation_verb",
        "compound", "duration", "distance"
    },
    "inference": {
        "causal", "conditional", "counterfact", "abductive",
        "normative", "predictive", "evaluative", "attribution",
    },
    "explicit_motivation": {
        "fear", "care", "distress", "protective", "control",
        "resource", "survival", "norm", "justification",
        "blame", "moral", "internalized"
    },
    "rational_advice": {
        "safety", "vulnerability", "action", "trigger", "resource",
        "retaliation", "contingency", "signal", "fallback",
        "exit", "support"
    }
}
