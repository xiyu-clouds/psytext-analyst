"""
🌊 全局常量池
"""
from pathlib import Path
from typing import Dict, Any, Final, List, Tuple

# ======================================================================
# 🌐 根目录（唯一真实源）
# ======================================================================
from src.state_of_mind.utils.data_validator import IS_DICT, IS_STR, IS_LIST, IS_FLOAT, IS_INT, IS_BOOL

ROOT_DIR: Final[Path] = Path(__file__).parent.parent.parent.parent.resolve()

# ======================================================================
# 📂 目录名称（字符串常量，用于组合）
# ======================================================================
# 根级目录
DIR_DATA = "data"
DIR_STATIC = "static"
DIR_XINJING = "src/state_of_mind"
# 默认输出根目录（可被 XINJING_OUTPUT_ROOT 覆盖）
DEFAULT_OUTPUT_ROOT: Final[Path] = ROOT_DIR / DIR_DATA

# 子模块目录
DIR_LOGS = "logs"
DIR_LOGS_FALLBACK = "logs_fallback"
DIR_YUAN = "yuan"
DIR_RAW = "raw"
DIR_DYE_VAT = "dye_vat"
DIR_CONFIG = "config"
DIR_OTHER = "other"
DIR_PROMPTS = "prompts"
DIR_TEMPLATES = "templates"

# ======================================================================
# 🛤️ 路径片段（Path 类型！不再是字符串）
# ======================================================================
# 使用 Path 对象统一管理路径，支持自然拼接 /
PATH_DATA = ROOT_DIR / DIR_DATA
PATH_STATIC = ROOT_DIR / DIR_STATIC
PATH_XINJING = ROOT_DIR / DIR_XINJING

# —————— 静态资源 ——————
PATH_STATIC_CONFIG = PATH_STATIC / DIR_CONFIG
PATH_STATIC_OTHER = PATH_STATIC / DIR_OTHER
PATH_STATIC_PROMPTS = PATH_STATIC / DIR_PROMPTS
PATH_STATIC_TEMPLATES = PATH_STATIC / DIR_TEMPLATES

# ======================================================================
# 📄 文件名（FILE_）——仍为字符串（文件名本身不含路径）
# ======================================================================
FILE_CONSTANTS = "constants.py"
FILE_ENUMS = "enums.py"
FILE_PROMPTS = "prompt.py"
FILE_PYPROJECT = "pyproject.toml"
FILE_DEFAULT_TEMPLATE = "default_template.html"
FILE_CHAINA_IP_LIST = "china_ip_list.txt"
FILE_APP_JSON = "app.json"

# ======================================================================
# 📄 完整文件路径（基于前面路径 + 文件名拼接而成）
# ======================================================================
PATH_FILE_PROMPTS = PATH_STATIC_PROMPTS / FILE_PROMPTS
PATH_FILE_APP_JSON = PATH_STATIC_CONFIG / FILE_APP_JSON

PATH_FILE_PYPROJECT = ROOT_DIR / FILE_PYPROJECT

# 中国IP文件路径
PATH_FILE_CHAINA_IP_LIST = PATH_STATIC_OTHER / FILE_CHAINA_IP_LIST
# 默认模板
PATH_FILE_DEFAULT_TEMPLATE = PATH_STATIC_TEMPLATES / FILE_DEFAULT_TEMPLATE

# ======================================================================
# 📄 日志路径
# ======================================================================
LOG_KEEP_DAYS = 7
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 10
PATH_ROOT_LOGS = PATH_DATA / DIR_LOGS
PATH_ROOT_LOGS_FALLBACK = PATH_DATA / DIR_LOGS_FALLBACK

# ======================================================================
# 🧩 枚举型常量（保持不变）
# ======================================================================
# 💾 存储后端
STORAGE_LOCAL = "local"
STORAGE_REDIS = "redis"

EVENT_RAW = "raw"
SUPPORTED_CATEGORIES = {EVENT_RAW}

# 模板分类
CATEGORY_RAW = "raw"
CATEGORY_SUGGESTION = "suggestion"


# 建议
class SuggestionType:
    PSYCHOANALYSIS = "psychoanalysis"
    CONSISTENCY_SUGGESTION = "consistency_suggestion"
    LITERARY_CRITIC = "literary_critic"
    IRONIC_DECONSTRUCTOR = "ironic_deconstructor"
    CRITICAL_THEORIST = "critical_theorist"
    EXISTENTIAL_PHILOSOPHER = "existential_philosopher"
    CULTURAL_ANTHROPOLOGIST = "cultural_anthropologist"


# 预处理 并行 串行
PREPROCESSING = "preprocessing"
PARALLEL = "parallel"
SERIAL = "serial"

COREFERENCE_RESOLUTION_BATCH = "coreference_resolution_batch"

# 大模型预处理
LLM_PARTICIPANTS_EXTRACTION = "LLM_PARTICIPANTS_EXTRACTION"

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
LLM_INFERENCE = "LLM_INFERENCE"
LLM_EXPLICIT_MOTIVATION_EXTRACTION = "LLM_EXPLICIT_MOTIVATION_EXTRACTION"
LLM_RATIONAL_ADVICE = "LLM_RATIONAL_ADVICE"

# 感知层常量集合，用于过滤非法参与者数据
PERCEPTION_LAYERS = {
    LLM_PERCEPTION_TEMPORAL_EXTRACTION,
    LLM_PERCEPTION_SPATIAL_EXTRACTION,
    LLM_PERCEPTION_VISUAL_EXTRACTION,
    LLM_PERCEPTION_AUDITORY_EXTRACTION,
    LLM_PERCEPTION_OLFACTORY_EXTRACTION,
    LLM_PERCEPTION_TACTILE_EXTRACTION,
    LLM_PERCEPTION_GUSTATORY_EXTRACTION,
    LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION,
    LLM_PERCEPTION_COGNITIVE_EXTRACTION,
    LLM_PERCEPTION_BODILY_EXTRACTION,
    LLM_PERCEPTION_EMOTIONAL_EXTRACTION,
    LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION,
}
# 定义各阶段并行任务允许使用的上下文 marker
ALLOWED_PARALLEL_MARKERS = {
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

# 定义各阶段串行任务允许使用的上下文 marker
ALLOWED_SERIAL_MARKERS = {
    0: {  # 第一步：合理推演
        "### PARTICIPANTS_VALID_INFORMATION BEGIN",
        "### PERCEPTUAL_CONTEXT_BATCH BEGIN",
        "### LEGITIMATE_PARTICIPANTS BEGIN"
    },
    1: {  # 第二步：显性动机
        "### PARTICIPANTS_VALID_INFORMATION BEGIN",
        "### PERCEPTUAL_CONTEXT_BATCH BEGIN",
        "### INFERENCE_CONTEXT BEGIN",
        "### LEGITIMATE_PARTICIPANTS BEGIN"
    },
    2: {  # 第三步：合理建议
        "### INFERENCE_CONTEXT BEGIN",
        "### EXPLICIT_MOTIVATION_CONTEXT BEGIN",
        "### LEGITIMATE_PARTICIPANTS BEGIN"
    }
}

# 语义模块常量（L1 判定依据）
SEMANTIC_MODULES_L1 = {
    "auditory", "visual", "olfactory", "cognitive", "interoceptive", "bodily",
    "social_relation", "temporal", "spatial", "tactile", "gustatory", "emotional"
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

# ======================================================================
# 🏷️ 模型能力标签
# ======================================================================


class ModelCapability:
    JSON_FORMAT = "json_format"
    REASONING = "reasoning"
    CODE = "code"
    VISION = "vision"
    AUDIO = "audio"
    EMOTION = "emotion"
    STREAMING = "streaming"


# ======================================================================
# 🧠 模型名称枚举
# ======================================================================
class ModelName:
    QWEN = "qwen"
    QWEN_MAX = "qwen-max"
    QWEN3_MAX = "qwen3-max"
    QWEN_PLUS = "qwen-plus"
    QWEN_FLASH = "qwen-flash"

    DEEPSEEK = "deepseek"
    DEEPSEEK_CHAT = "deepseek-chat"


# ======================================================================
# 📘 模型配置元信息（MODEL_CONFIG）
# ======================================================================
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    ModelName.QWEN3_MAX: {
        "provider": "qwen",
        "description": "Qwen3 系列最强模型，复杂推理、多步骤任务首选。",
        "doc_url": "https://help.aliyun.com/zh/model-studio/developer-reference/qwen3-max",
        "recommended_params": {
            "temperature": 0.6,
            "top_p": 0.8,
            "max_output_tokens": 4096,
            "result_format": "message"
        },
        "capabilities": {
            ModelCapability.JSON_FORMAT: True,
            ModelCapability.REASONING: True,
            ModelCapability.CODE: True,
            ModelCapability.VISION: False,
            ModelCapability.AUDIO: False,
            ModelCapability.EMOTION: False,
            ModelCapability.STREAMING: True,
        }
    },
    ModelName.QWEN_PLUS: {
        "provider": "qwen",
        "description": "性能与成本均衡，适用于中高复杂度任务。",
        "doc_url": "https://help.aliyun.com/zh/model-studio/developer-reference/qwen-plus",
        "recommended_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 1024,
            "result_format": "json_object"
        },
        "capabilities": {
            ModelCapability.JSON_FORMAT: True,
            ModelCapability.REASONING: True,
            ModelCapability.CODE: True,
            ModelCapability.VISION: False,
            ModelCapability.AUDIO: False,
            ModelCapability.EMOTION: False,
            ModelCapability.STREAMING: True,
        }
    },
    ModelName.QWEN_MAX: {
        "provider": "qwen",
        "description": "Qwen2 最强通用模型（逐步被 qwen3 替代）。",
        "doc_url": "https://help.aliyun.com/zh/model-studio/developer-reference/qwen-max",
        "recommended_params": {
            "temperature": 0.6,
            "top_p": 0.8,
            "max_output_tokens": 1024,
            "result_format": "json_object"
        },
        "capabilities": {
            ModelCapability.JSON_FORMAT: True,
            ModelCapability.REASONING: True,
            ModelCapability.CODE: True,
            ModelCapability.VISION: False,
            ModelCapability.AUDIO: False,
            ModelCapability.EMOTION: False,
            ModelCapability.STREAMING: True,
        }
    },
    ModelName.QWEN_FLASH: {
        "provider": "qwen",
        "description": "极速轻量模型，适合高并发实时对话。",
        "doc_url": "https://help.aliyun.com/zh/model-studio/developer-reference/qwen3-flash",
        "recommended_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 4096,
            "result_format": "message"
        },
        "capabilities": {
            ModelCapability.JSON_FORMAT: True,
            ModelCapability.REASONING: True,
            ModelCapability.CODE: True,
            ModelCapability.VISION: False,
            ModelCapability.AUDIO: False,
            ModelCapability.EMOTION: False,
            ModelCapability.STREAMING: True,
        }
    },
    # ModelName.DEEPSEEK_REASONER: {
    #     "provider": "deepseek",
    #     "description": "专为复杂逻辑与数学推导优化的推理模型。",
    #     "doc_url": "https://platform.deepseek.com/api-docs/models/deepseek-reasoner",
    #     "recommended_params": {
    #         "temperature": 0.3,
    #         "top_p": 0.5,
    #         "max_tokens": 4096
    #     },
    #     "capabilities": {
    #         ModelCapability.JSON_FORMAT: False,
    #         ModelCapability.REASONING: True,
    #         ModelCapability.CODE: True,
    #         ModelCapability.VISION: False,
    #         ModelCapability.AUDIO: False,
    #         ModelCapability.EMOTION: False,
    #         ModelCapability.STREAMING: True,
    #     }
    # },
    ModelName.DEEPSEEK_CHAT: {
        "provider": "deepseek",
        "description": "通用对话模型，流畅交互与代码生成。",
        "doc_url": "https://platform.deepseek.com/api-docs/models/deepseek-chat",
        "recommended_params": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        },
        "capabilities": {
            ModelCapability.JSON_FORMAT: True,
            ModelCapability.REASONING: False,
            ModelCapability.CODE: True,
            ModelCapability.VISION: False,
            ModelCapability.AUDIO: False,
            ModelCapability.EMOTION: False,
            ModelCapability.STREAMING: True,
        }
    }
}

"""校验数据结构,只有顶层是列表的时候子级才加通配符"""
REQUIRED_FIELDS_BY_CATEGORY: Dict[str, Dict[str, List[Tuple[str, bool, Any, str]]]] = {
    CATEGORY_RAW: {
        # ────────────────────────────────────────
        # 1. 源数据提取（参与者列表）—— 你已确认，保留不变
        # ────────────────────────────────────────
        LLM_PARTICIPANTS_EXTRACTION: [
            # === 核心标识 ===
            ("participants", True, IS_LIST, "participants（参与者列表）："),
            ("participants.*.entity", True, IS_STR, "entity（原始指称短语）："),

            # === 静态社会属性 ===
            ("participants.*.social_role", False, IS_STR, "social_role（社会角色）："),
            ("participants.*.occupation", False, IS_STR, "occupation（职业身份）："),
            ("participants.*.family_status ", False, IS_STR, "family_status （家庭状态）："),
            ("participants.*.education_level", False, IS_STR, "education_level（教育程度）："),
            ("participants.*.cultural_identity", False, IS_LIST, "cultural_identity（文化身份标签）："),
            ("participants.*.primary_language", False, IS_STR, "primary_language（主要使用语言）："),
            ("participants.*.institutional_affiliation", False, IS_LIST, "institutional_affiliation（所属机构）："),
            ("participants.*.beliefs_or_values", False, IS_STR, "beliefs_or_values（信仰或价值观）："),

            # === 生物与生理属性 ===
            ("participants.*.age_range", False, IS_STR, "age_range（年龄范围）："),
            ("participants.*.gender", False, IS_STR, "gender（性别或相关表述）："),
            ("participants.*.ethnicity_or_origin", False, IS_STR, "ethnicity_or_origin（族群/国籍/地域出身）："),
            ("participants.*.physical_traits", False, IS_LIST, "physical_traits（固有生理特征）："),
            ("participants.*.current_physical_state", False, IS_STR, "current_physical_state（当前身体状态）："),
            ("participants.*.visible_injury_or_wound", False, IS_LIST, "visible_injury_or_wound（可见伤痕或包扎）："),

            # === 感官与外显特征 ===
            ("participants.*.appearance", False, IS_LIST, "appearance（外貌、衣着、姿态或表情）："),
            ("participants.*.voice_quality", False, IS_STR, "voice_quality（嗓音特质）："),
            ("participants.*.inherent_odor", False, IS_LIST, "inherent_odor（固有体味）："),
            ("participants.*.carried_objects", False, IS_LIST, "carried_objects（持有物品）："),
            ("participants.*.worn_technology", False, IS_LIST, "worn_technology（佩戴的电子设备）："),

            # === 心理与行为属性 ===
            ("participants.*.personality_traits", False, IS_LIST, "personality_traits（长期人格特质）："),
            ("participants.*.behavioral_tendencies", False, IS_LIST, "behavioral_tendencies（稳定行为倾向）："),
            ("participants.*.speech_pattern", False, IS_STR, "speech_pattern（说话方式）："),

            # === 交互与场景角色 ===
            ("participants.*.interaction_role", False, IS_STR, "interaction_role（当前互动角色）："),
        ],

        # ────────────────────────────────────────
        # 2. 时间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TEMPORAL_EXTRACTION: [
            ("temporal", False, IS_DICT, "temporal（时间感知根对象）："),
            ("temporal.events", False, IS_LIST, "events（时间事件列表）："),
            ("temporal.evidence", False, IS_LIST, "evidence（支撑整体时间判断的原文片段）："),
            ("temporal.summary", False, IS_STR, "summary（客观提取生成的时间情景摘要）："),

            ("temporal.events.*.experiencer", False, IS_STR, "experiencer（时间事件的感知或陈述主体）："),
            ("temporal.events.*.evidence", False, IS_LIST, "evidence（支撑时间判断的原始文本片段）："),
            ("temporal.events.*.semantic_notation", False, IS_STR, "semantic_notation（时间事件的语义标识）："),
            ("temporal.events.*.exact_literals", False, IS_LIST, "exact_literals（原文中显式出现的精确时间字面量）："),
            ("temporal.events.*.relative_expressions", False, IS_LIST, "relative_expressions（原文中的相对或模糊时间表达）："),
            ("temporal.events.*.negated_time", False, IS_LIST, "negated_time（被显式否定的时间表达）："),
            ("temporal.events.*.time_ranges", False, IS_LIST, "time_ranges（原文中出现的时间区间）："),
            ("temporal.events.*.durations", False, IS_LIST, "durations（原文中提及的持续时间表达）："),
            ("temporal.events.*.frequencies", False, IS_LIST, "frequencies（原文中出现的周期性或频率表达）："),
            ("temporal.events.*.event_markers", False, IS_LIST, "event_markers（与时间共现的事件关键词）："),
            ("temporal.events.*.tense_aspect", False, IS_STR, "tense_aspect（显式时体标记）："),
            ("temporal.events.*.seasonal_or_cultural_time", False, IS_LIST, "seasonal_or_cultural_time（节日、节气、财季等）："),
            ("temporal.events.*.temporal_anchor", False, IS_STR, "temporal_anchor（显式时间参考锚点）："),
            ("temporal.events.*.uncertainty_modifiers", False, IS_LIST, "uncertainty_modifiers（时间不确定性修饰语）：")
        ],

        # ────────────────────────────────────────
        # 3. 空间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_SPATIAL_EXTRACTION: [
            ("spatial", False, IS_DICT, "spatial（空间感知根对象）："),
            ("spatial.events", False, IS_LIST, "events（空间事件列表）："),
            ("spatial.evidence", False, IS_LIST, "evidence（支撑整体空间判断的原文片段）："),
            ("spatial.summary", False, IS_STR, "summary（客观提取生成的空间情景摘要）："),

            ("spatial.events.*.experiencer", False, IS_STR, "experiencer（空间描述的感知或陈述主体）："),
            ("spatial.events.*.evidence", False, IS_LIST, "evidence（支撑空间判断的原始文本片段）："),
            ("spatial.events.*.semantic_notation", False, IS_STR, "semantic_notation（空间事件的标准化语义标识）："),
            ("spatial.events.*.places", False, IS_LIST, "places（原文中提及的具体地点或场所名称）："),
            ("spatial.events.*.layout_descriptions", False, IS_LIST, "layout_descriptions（原文中对空间结构或布局的描述）："),
            ("spatial.events.*.negated_places", False, IS_LIST, "negated_places（显式否定的地点）："),
            ("spatial.events.*.spatial_event_markers", False, IS_LIST, "spatial_event_markers（与地点共现的事件动词/名词）："),
            ("spatial.events.*.cultural_or_institutional_spaces", False, IS_LIST,
             "cultural_or_institutional_spaces（制度性/文化性空间单位）："),
            ("spatial.events.*.orientation_descriptions", False, IS_LIST, "orientation_descriptions（明确的方向或朝向描述）："),

            ("spatial.events.*.proximity_relations", False, IS_LIST, "proximity_relations（空间参与者之间的关系实例列表）："),
            ("spatial.events.*.proximity_relations.*.actor", False, IS_STR, "actor（空间关系中的主动方或参照主体）："),
            ("spatial.events.*.proximity_relations.*.target", False, IS_STR, "target（空间关系中的目标方或被参照对象）："),
            ("spatial.events.*.proximity_relations.*.distance_cm", False, IS_INT, "distance_cm（若原文明确提及，以厘米为单位的物理距离）："),
            ("spatial.events.*.proximity_relations.*.modifiers", False, IS_LIST, "modifiers（修饰性空间成分）："),
            ("spatial.events.*.proximity_relations.*.relation_type", False, IS_STR, "relation_type（空间关系类型）：")
        ],

        # ────────────────────────────────────────
        # 4. 视觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_VISUAL_EXTRACTION: [
            ("visual", False, IS_DICT, "visual（视觉感知根对象）："),
            ("visual.events", False, IS_LIST, "events（视觉事件列表）："),
            ("visual.evidence", False, IS_LIST, "evidence（支撑整体视觉判断的原文片段）："),
            ("visual.summary", False, IS_STR, "summary（基于客观提取生成的视觉情景摘要）："),

            ("visual.events.*.experiencer", False, IS_STR, "experiencer（观察主体）："),
            ("visual.events.*.evidence", False, IS_LIST, "evidence（支撑该观察的原文片段）："),
            ("visual.events.*.semantic_notation", False, IS_STR, "semantic_notation（该视觉事件的标准化语义标识）："),
            ("visual.events.*.observed_entity", False, IS_STR, "observed_entity（被观察的对象或主体）："),
            ("visual.events.*.visual_objects", False, IS_LIST, "visual_objects（原文中明确提及的可见物体）："),
            ("visual.events.*.visual_attributes", False, IS_LIST, "visual_attributes（对象的视觉属性）："),
            ("visual.events.*.visual_actions", False, IS_LIST, "visual_actions（可见的动作或姿态）："),
            ("visual.events.*.gaze_target", False, IS_STR, "gaze_target（注视目标）："),
            ("visual.events.*.eye_contact", False, IS_LIST, "eye_contact（眼神交互描述）："),
            ("visual.events.*.facial_cues", False, IS_LIST, "facial_cues（面部表情或微表情线索）："),
            ("visual.events.*.salience", False, IS_FLOAT, "salience（该视觉观察的显著性或确定性）："),
            ("visual.events.*.negated_observations", False, IS_LIST, "negated_observations（显式否定的视觉行为）："),
            ("visual.events.*.visual_medium", False, IS_LIST, "visual_medium（视觉依赖的媒介）："),
            ("visual.events.*.occlusion_or_obstruction", False, IS_LIST, "occlusion_or_obstruction（明确提及的遮挡物或视线阻碍）："),
            ("visual.events.*.lighting_conditions", False, IS_LIST, "lighting_conditions（显式描述的光照条件）："),
        ],

        # ────────────────────────────────────────
        # 5. 听觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_AUDITORY_EXTRACTION: [
            ("auditory", False, IS_DICT, "auditory（听觉感知根对象）："),
            ("auditory.events", False, IS_LIST, "events（听觉事件列表）："),
            ("auditory.evidence", False, IS_LIST, "evidence（支撑整体听觉判断的原文片段）："),
            ("auditory.summary", False, IS_STR, "summary（基于客观提取生成的听觉情景摘要）："),

            ("auditory.events.*.experiencer", False, IS_STR, "experiencer（听觉接收主体）："),
            ("auditory.events.*.evidence", False, IS_LIST, "evidence（支撑该听觉事件的原文片段）："),
            ("auditory.events.*.semantic_notation", False, IS_STR, "semantic_notation（该听觉事件的标准化语义标识）："),
            ("auditory.events.*.sound_source", False, IS_STR, "sound_source（发声主体或声源）："),
            ("auditory.events.*.auditory_content", False, IS_LIST, "auditory_content（直接描述的听觉内容关键词或原文片段）："),
            ("auditory.events.*.is_primary_focus", False, IS_BOOL, "is_primary_focus（是否为当前听觉焦点）："),
            ("auditory.events.*.prosody_cues", False, IS_LIST, "prosody_cues（直接描述的声音特征）："),
            ("auditory.events.*.pause_description", False, IS_STR, "pause_description（明确描述的停顿特征）："),
            ("auditory.events.*.intensity", False, IS_FLOAT, "intensity（听觉感知强度，基于修饰词量化）："),
            ("auditory.events.*.negated_observations", False, IS_LIST, "negated_observations（显式否定的听觉行为）："),
            ("auditory.events.*.auditory_medium", False, IS_LIST, "auditory_medium（听觉依赖的媒介）："),
            ("auditory.events.*.background_sounds", False, IS_LIST, "background_sounds（明确提及的环境声或背景噪音）："),
            ("auditory.events.*.nonverbal_sounds", False, IS_LIST, "nonverbal_sounds（非语言声音）：")
        ],

        # ────────────────────────────────────────
        # 6. 嗅觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_OLFACTORY_EXTRACTION: [
            ("olfactory", False, IS_DICT, "olfactory（嗅觉感知根对象）："),
            ("olfactory.events", False, IS_LIST, "events（嗅觉事件列表）："),
            ("olfactory.evidence", False, IS_LIST, "evidence（支撑整体嗅觉判断的原文片段）："),
            ("olfactory.summary", False, IS_STR, "summary（基于客观提取生成的嗅觉情景摘要）："),

            ("olfactory.events.*.experiencer", False, IS_STR, "experiencer（气味感知主体）："),
            ("olfactory.events.*.evidence", False, IS_LIST, "evidence（支撑该嗅觉事件的原文片段）："),
            ("olfactory.events.*.semantic_notation", False, IS_STR, "semantic_notation（该嗅觉事件的标准化语义标识）："),
            ("olfactory.events.*.odor_source", False, IS_STR, "odor_source（气味来源主体或对象）："),
            ("olfactory.events.*.odor_descriptors", False, IS_LIST, "odor_descriptors（直接出现的气味描述词或短语）："),
            ("olfactory.events.*.intensity", False, IS_FLOAT, "intensity（嗅觉感知强度，基于修饰词量化）："),
            ("olfactory.events.*.negated_observations", False, IS_LIST, "negated_observations（显式否定的嗅觉行为）："),
            ("olfactory.events.*.odor_valence", False, IS_LIST, "odor_valence（气味的情感/评价词）："),
            ("olfactory.events.*.odor_source_category", False, IS_LIST, "odor_source_category（气味来源的大类）："),
            ("olfactory.events.*.olfactory_actions", False, IS_LIST, "olfactory_actions（与嗅觉相关的身体动作）：")
        ],

        # ────────────────────────────────────────
        # 7. 触觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TACTILE_EXTRACTION: [
            ("tactile", False, IS_DICT, "tactile（触觉感知根对象）："),
            ("tactile.events", False, IS_LIST, "events（触觉事件列表）："),
            ("tactile.evidence", False, IS_LIST, "evidence（支撑整体触觉判断的原文片段）："),
            ("tactile.summary", False, IS_STR, "summary（基于客观提取生成的触觉情景摘要）："),

            ("tactile.events.*.experiencer", False, IS_STR, "experiencer（触觉体验主体）："),
            ("tactile.events.*.evidence", False, IS_LIST, "evidence（支撑该触觉事件的原文片段）："),
            ("tactile.events.*.semantic_notation", False, IS_STR, "semantic_notation（该触觉事件的标准化语义标识）："),
            ("tactile.events.*.contact_target", False, IS_STR, "contact_target（被接触对象或身体部位）："),
            ("tactile.events.*.tactile_descriptors", False, IS_LIST, "tactile_descriptors（直接描述的触觉感受或动作）："),
            ("tactile.events.*.intensity", False, IS_FLOAT, "intensity（触觉感知强度，基于修饰词量化）："),
            ("tactile.events.*.contact_initiator", False, IS_STR, "contact_initiator（主动发起接触的一方）："),
            ("tactile.events.*.body_part", False, IS_STR, "body_part（触觉发生的身体部位）："),
            ("tactile.events.*.texture", False, IS_LIST, "texture（质地类描述）："),
            ("tactile.events.*.temperature", False, IS_LIST, "temperature（温度类描述）："),
            ("tactile.events.*.pressure", False, IS_LIST, "pressure（压力类描述）："),
            ("tactile.events.*.pain", False, IS_LIST, "pain（疼痛类描述）："),
            ("tactile.events.*.motion", False, IS_LIST, "motion（动态触觉描述）："),
            ("tactile.events.*.vibration", False, IS_LIST, "vibration（震动类描述）："),
            ("tactile.events.*.moisture", False, IS_LIST, "moisture（湿度/干湿类描述）："),
            ("tactile.events.*.contact", False, IS_LIST, "contact（接触存在性或方式描述）："),
            ("tactile.events.*.negated_observations", False, IS_LIST, "negated_observations（显式否定的触觉）："),
            ("tactile.events.*.tactile_intent_or_valence", False, IS_LIST, "tactile_intent_or_valence（触觉的情感或意图词）：")
        ],

        # ────────────────────────────────────────
        # 8. 味觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_GUSTATORY_EXTRACTION: [
            ("gustatory", False, IS_DICT, "gustatory（味觉感知根对象）："),
            ("gustatory.events", False, IS_LIST, "events（味觉事件列表）："),
            ("gustatory.evidence", False, IS_LIST, "evidence（支撑整体味觉判断的原文片段）："),
            ("gustatory.summary", False, IS_STR, "summary（基于客观提取生成的味觉情景摘要）："),

            ("gustatory.events.*.experiencer", False, IS_STR, "experiencer（味觉体验主体）："),
            ("gustatory.events.*.evidence", False, IS_LIST, "evidence（支撑该味觉事件的原文片段）："),
            ("gustatory.events.*.semantic_notation", False, IS_STR, "semantic_notation（该味觉事件的标准化语义标识）："),
            ("gustatory.events.*.intensity", False, IS_FLOAT, "intensity（味觉感知强度，基于修饰词量化）："),
            ("gustatory.events.*.taste_source", False, IS_STR, "taste_source（食物或味道来源）："),
            ("gustatory.events.*.taste_descriptors", False, IS_LIST, "taste_descriptors（直接描述的味道或短语）："),
            ("gustatory.events.*.contact_initiator", False, IS_STR, "contact_initiator（主动摄入者）："),
            ("gustatory.events.*.body_part", False, IS_STR, "body_part（味觉发生部位）："),
            ("gustatory.events.*.intent_or_valence", False, IS_LIST, "tactile_intent_or_valence（味觉的情感或意图词）："),
            ("gustatory.events.*.negated_observations", False, IS_LIST, "negated_observations（显式否定的味觉）："),
            ("gustatory.events.*.sweet", False, IS_LIST, "sweet（甜味类描述）："),
            ("gustatory.events.*.salty", False, IS_LIST, "salty（咸味类描述）："),
            ("gustatory.events.*.sour", False, IS_LIST, "sour（酸味类描述）："),
            ("gustatory.events.*.bitter", False, IS_LIST, "bitter（苦味类描述）："),
            ("gustatory.events.*.umami", False, IS_LIST, "umami（鲜味类描述）："),
            ("gustatory.events.*.spicy", False, IS_LIST, "spicy（辣味/刺激类描述）："),
            ("gustatory.events.*.astringent", False, IS_LIST, "astringent（涩味类描述）："),
            ("gustatory.events.*.fatty", False, IS_LIST, "fatty（油脂感描述）："),
            ("gustatory.events.*.metallic", False, IS_LIST, "metallic（金属味描述）："),
            ("gustatory.events.*.chemical", False, IS_LIST, "chemical（化学异味）："),
            ("gustatory.events.*.thermal", False, IS_LIST, "thermal（化学性冷热感描述）：")
        ],

        # ────────────────────────────────────────
        # 9. 内感受
        # ────────────────────────────────────────
        LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION: [
            ("interoceptive", False, IS_DICT, "interoceptive（内感受感知根对象）："),
            ("interoceptive.events", False, IS_LIST, "events（内感受事件列表）："),
            ("interoceptive.evidence", False, IS_LIST, "evidence（支撑整体内感受判断的原文片段）："),
            ("interoceptive.summary", False, IS_STR, "summary（基于客观提取生成的内感受情景摘要）："),

            ("interoceptive.events.*.experiencer", False, IS_STR, "experiencer（主观感受的体验者）："),
            ("interoceptive.events.*.intensity", False, IS_FLOAT, "intensity（内感受强度，基于修饰词量化）："),
            ("interoceptive.events.*.evidence", False, IS_LIST, "evidence（支撑该内感受事件的原文片段）："),
            ("interoceptive.events.*.semantic_notation", False, IS_STR, "semantic_notation（该内感受事件的标准化语义标识）："),
            ("interoceptive.events.*.contact_initiator", False, IS_STR, "contact_initiator（内感受触发者）："),
            ("interoceptive.events.*.body_part", False, IS_STR, "body_part（感受发生部位）："),
            ("interoceptive.events.*.intent_or_valence", False, IS_LIST,
             "tactile_intent_or_valence（内感受的情感或意图词）："),
            ("interoceptive.events.*.negated_observations", False, IS_LIST, "negated_observations（显式否定的内感受）："),
            ("interoceptive.events.*.cardiac", False, IS_LIST, "cardiac（心悸/心跳类描述）："),
            ("interoceptive.events.*.respiratory", False, IS_LIST, "respiratory（呼吸类描述）："),
            ("interoceptive.events.*.gastrointestinal", False, IS_LIST, "gastrointestinal（胃肠类描述）："),
            ("interoceptive.events.*.thermal", False, IS_LIST, "thermal（体温/冷热感）："),
            ("interoceptive.events.*.muscular", False, IS_LIST, "muscular（肌肉紧张/酸痛）："),
            ("interoceptive.events.*.visceral_pressure", False, IS_LIST, "visceral_pressure（压迫感）："),
            ("interoceptive.events.*.dizziness", False, IS_LIST, "dizziness（眩晕/失衡感）："),
            ("interoceptive.events.*.nausea", False, IS_LIST, "nausea（恶心/反胃感）："),
            ("interoceptive.events.*.fatigue", False, IS_LIST, "fatigue（疲惫/虚脱感）："),
            ("interoceptive.events.*.thirst_hunger", False, IS_LIST, "thirst_hunger（饥渴感）：")
        ],

        # ────────────────────────────────────────
        # 10. 认知过程
        # ────────────────────────────────────────
        LLM_PERCEPTION_COGNITIVE_EXTRACTION: [
            ("cognitive", False, IS_DICT, "cognitive（认知过程根对象）："),
            ("cognitive.events", False, IS_LIST, "events（认知事件列表）："),
            ("cognitive.evidence", False, IS_LIST, "evidence（支撑整体认知判断的原文片段）："),
            ("cognitive.summary", False, IS_STR, "summary（基于客观提取生成的认知情景摘要）："),

            ("cognitive.events.*.experiencer", False, IS_STR, "experiencer（认知主体）："),
            ("cognitive.events.*.intensity", False, IS_FLOAT, "intensity（认知负荷或确信强度）："),
            ("cognitive.events.*.evidence", False, IS_LIST, "evidence（支撑该认知事件的原文片段）："),
            ("cognitive.events.*.semantic_notation", False, IS_STR, "semantic_notation（该认知事件的标准化语义标识）："),
            ("cognitive.events.*.cognitive_agent", False, IS_STR, "cognitive_agent（思维发起者）："),
            ("cognitive.events.*.target_entity", False, IS_STR, "target_entity（思维指向的对象或主题）："),
            ("cognitive.events.*.cognitive_valence", False, IS_LIST, "cognitive_valence（认知情感倾向词）："),
            ("cognitive.events.*.negated_cognitions", False, IS_LIST, "negated_cognitions（显式否定的认知）："),
            ("cognitive.events.*.belief", False, IS_LIST, "belief（信念陈述）："),
            ("cognitive.events.*.intention", False, IS_LIST, "intention（意图表达）："),
            ("cognitive.events.*.inference", False, IS_LIST, "inference（推理过程）："),
            ("cognitive.events.*.memory_recall", False, IS_LIST, "memory_recall（记忆提取）："),
            ("cognitive.events.*.doubt_or_uncertainty", False, IS_LIST, "doubt_or_uncertainty（怀疑/不确定）："),
            ("cognitive.events.*.evaluation", False, IS_LIST, "evaluation（价值判断）："),
            ("cognitive.events.*.problem_solving", False, IS_LIST, "problem_solving（问题解决思路）："),
            ("cognitive.events.*.metacognition", False, IS_LIST, "metacognition（元认知）：")
        ],

        # ────────────────────────────────────────
        # 11. 躯体化表现
        # ────────────────────────────────────────
        LLM_PERCEPTION_BODILY_EXTRACTION: [
            ("bodily", False, IS_DICT, "bodily（躯体化表现根对象）："),
            ("bodily.events", False, IS_LIST, "events（躯体化事件列表）："),
            ("bodily.evidence", False, IS_LIST, "evidence（支撑整体躯体化判断的原文片段）："),
            ("bodily.summary", False, IS_STR, "summary（基于客观提取生成的躯体化情景摘要）："),

            ("bodily.events.*.experiencer", False, IS_STR, "experiencer（躯体行为主体）："),
            ("bodily.events.*.intensity", False, IS_FLOAT, "intensity（躯体化表现强度）："),
            ("bodily.events.*.evidence", False, IS_LIST, "evidence（支撑该躯体化事件的原文片段）："),
            ("bodily.events.*.semantic_notation", False, IS_STR, "semantic_notation（该躯体化事件的标准化语义标识）："),
            ("bodily.events.*.observer", False, IS_STR, "observer（观察者）："),
            ("bodily.events.*.movement_direction", False, IS_STR, "movement_direction（运动方向）："),
            ("bodily.events.*.posture", False, IS_STR, "posture（静态姿态）："),
            ("bodily.events.*.facial_expression", False, IS_LIST, "facial_expression（面部表情显式描述）："),
            ("bodily.events.*.vocal_behavior", False, IS_LIST, "vocal_behavior（声音相关躯体表现）："),
            ("bodily.events.*.autonomic_signs", False, IS_LIST, "autonomic_signs（自主神经外显征象）："),
            ("bodily.events.*.motor_behavior", False, IS_LIST, "motor_behavior（随意运动行为）："),
            ("bodily.events.*.freeze_or_faint", False, IS_LIST, "freeze_or_faint（冻结/晕厥类反应）：")
        ],

        # ────────────────────────────────────────
        # 12. 情感状态
        # ────────────────────────────────────────
        LLM_PERCEPTION_EMOTIONAL_EXTRACTION: [
            ("emotional", False, IS_DICT, "emotional（情感状态根对象）："),
            ("emotional.events", False, IS_LIST, "events（情感事件列表）："),
            ("emotional.evidence", False, IS_LIST, "evidence（支撑整体情感判断的原文片段）："),
            ("emotional.summary", False, IS_STR, "summary（基于客观提取生成的情感情景摘要）："),

            ("emotional.events.*.experiencer", False, IS_STR, "experiencer（情绪主体）："),
            ("emotional.events.*.intensity", False, IS_FLOAT, "intensity（情感强度）："),
            ("emotional.events.*.evidence", False, IS_LIST, "evidence（支撑该情感事件的原文片段）："),
            ("emotional.events.*.semantic_notation", False, IS_STR, "semantic_notation（该情感事件的标准化语义标识）："),
            ("emotional.events.*.expression_mode", False, IS_STR, "expression_mode（情绪表达模式）："),
            ("emotional.events.*.emotion_labels", False, IS_LIST, "emotion_labels（显式情绪词或短语）："),
            ("emotional.events.*.valence", False, IS_FLOAT, "valence（情绪效价）："),
            ("emotional.events.*.arousal", False, IS_FLOAT, "arousal（情绪唤醒度）：")
        ],

        # ────────────────────────────────────────
        # 13. 社会关系
        # ────────────────────────────────────────
        LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION: [
            ("social_relation", False, IS_DICT, "social_relation（社会关系根对象）："),
            ("social_relation.events", False, IS_LIST, "events（社会关系事件列表）："),
            ("social_relation.evidence", False, IS_LIST, "evidence（支撑整体社会关系判断的原文片段）："),
            ("social_relation.summary", False, IS_STR, "summary（社会关系情景客观摘要）："),

            ("social_relation.events.*.experiencer", False, IS_STR, "experiencer（社会关系主体）："),
            ("social_relation.events.*.semantic_notation", False, IS_STR, "semantic_notation（标准化关系语义标识）："),
            ("social_relation.events.*.relation_type", False, IS_STR, "relation_type（自然语言描述的关系类型）："),
            ("social_relation.events.*.participants", False, IS_LIST, "participants（所有涉及者，用于对称关系）："),
            ("social_relation.events.*.source", False, IS_STR, "source（关系发起方/主语，用于非对称关系）："),
            ("social_relation.events.*.target", False, IS_STR, "target（关系接收方/宾语，用于非对称关系）："),
            ("social_relation.events.*.confidence", False, IS_FLOAT, "confidence（关系判断置信度）："),
            ("social_relation.events.*.evidence", False, IS_LIST, "evidence（该关系事件的原文证据）：")
        ],

        # ────────────────────────────────────────
        # 14. 合理推演
        # ────────────────────────────────────────
        LLM_INFERENCE: [
            ("inference", False, IS_DICT, "inference（推理层根对象）："),
            ("inference.events", False, IS_LIST, "events（具体推理事件列表）："),
            ("inference.evidence", False, IS_LIST, "evidence（支撑整体推理的原始文本依据片段列表）："),
            ("inference.summary", False, IS_STR, "summary（基于所有推理事件得出的整体情景性结论摘要）："),

            ("inference.events.*.experiencer", False, IS_STR, "experiencer（推理主体）："),
            ("inference.events.*.inference_type", False, IS_STR, "inference_type（推理类型）："),
            ("inference.events.*.anchor_points", False, IS_LIST, "anchor_points（所依赖的感知层事件的 semantic_notation 列表）："),
            ("inference.events.*.inferred_proposition", False, IS_STR, "inferred_proposition（用自然语言陈述的推理结论）："),
            ("inference.events.*.evidence", False, IS_LIST, "evidence（支撑该推理的原始文本片段列表）："),
            ("inference.events.*.semantic_notation", False, IS_STR, "semantic_notation（该推理事件的标准化语义标识）："),
            ("inference.events.*.polarity", False, IS_STR, "polarity（命题极性）："),
            ("inference.events.*.context_modality", False, IS_STR, "context_modality（语境模态）："),
            ("inference.events.*.scope", False, IS_STR, "scope（推理适用范围）：")
        ],

        # ────────────────────────────────────────
        # 15. 显性动机
        # ────────────────────────────────────────
        LLM_EXPLICIT_MOTIVATION_EXTRACTION: [
            ("explicit_motivation", False, IS_DICT, "explicit_motivation（显性动机根对象）："),
            ("explicit_motivation.events", False, IS_LIST, "events（显性动机事件列表）："),
            ("explicit_motivation.evidence", False, IS_LIST, "evidence（支撑显性动机的原始文本依据）："),
            ("explicit_motivation.summary", False, IS_STR, "summary（基于显性陈述整合的深层动因摘要）："),

            ("explicit_motivation.events.*.experiencer", False, IS_STR, "experiencer（显性动机的参与者主体对象）："),
            ("explicit_motivation.events.*.evidence", False, IS_LIST, "evidence（支持该显性动机的原始文本依据）："),
            ("explicit_motivation.events.*.semantic_notation", False, IS_STR, "semantic_notation（该显性动机的标准语意标识）："),
            ("explicit_motivation.events.*.core_driver", False, IS_LIST, "core_driver（用户明确表达的根本需求、恐惧或动机）："),
            ("explicit_motivation.events.*.care_expression", False, IS_LIST, "care_expression（明确表达的关怀行为或意图）："),
            ("explicit_motivation.events.*.separation_anxiety", False, IS_LIST,
             "separation_anxiety（因分离而显式陈述的担忧、恐惧或回忆）："),
            ("explicit_motivation.events.*.protective_intent", False, IS_LIST,
             "protective_intent（为对方健康、安全或福祉采取行动的直接表述）："),

            ("explicit_motivation.events.*.power_asymmetry", False, IS_DICT, "power_asymmetry（权力差异结构）："),
            ("explicit_motivation.events.*.power_asymmetry.control_axis", False, IS_LIST,
             "control_axis（明确提到的控制维度，如情感绑定、义务施加、资源条件化）："),
            ("explicit_motivation.events.*.power_asymmetry.threat_vector", False, IS_LIST,
             "threat_vector（直接陈述的威胁方式，如关系断裂、情感撤回）："),
            ("explicit_motivation.events.*.power_asymmetry.evidence", False, IS_LIST, "evidence（支撑权力分析的原文片段）："),

            ("explicit_motivation.events.*.resource_control", False, IS_LIST,
             "resource_control（明确指出对方掌控的关键资源，且以之作为交换或惩罚手段）："),
            ("explicit_motivation.events.*.survival_imperative", False, IS_LIST,
             "survival_imperative（亲口表达的服从理由，涉及基本生存、安全或稳定）："),
            ("explicit_motivation.events.*.social_enforcement_mechanism", False, IS_LIST,
             "social_enforcement_mechanism（提及的社会规范、家庭压力或群体期待）："),

            ("explicit_motivation.events.*.narrative_distortion", False, IS_DICT, "narrative_distortion（话术策略）："),
            ("explicit_motivation.events.*.narrative_distortion.self_justification", False, IS_STR,
             "self_justification（为自身行为提供的直接合理化语句）："),
            ("explicit_motivation.events.*.narrative_distortion.blame_shift", False, IS_STR, "blame_shift（明确转嫁责任的原话）："),
            ("explicit_motivation.events.*.narrative_distortion.moral_licensing", False, IS_STR,
             "moral_licensing（以道德身份豁免行为的原话）："),
            ("explicit_motivation.events.*.narrative_distortion.evidence", False, IS_LIST, "evidence（支撑话术分析的原文片段）："),
            ("explicit_motivation.events.*.internalized_oppression", False, IS_LIST,
             "internalized_oppression（用户自我贬低、正当化对方伤害或接受不公待遇的显式陈述）："),
            ("explicit_motivation.events.*.motivation_category", False, IS_STR, "motivation_category（该事件主导的显性动机类型标签）：")
        ],

        # ────────────────────────────────────────
        # 16. 合理建议
        # ────────────────────────────────────────
        LLM_RATIONAL_ADVICE: [
            ("rational_advice", False, IS_DICT, "rational_advice（合理建议根对象）："),
            ("rational_advice.summary", False, IS_STR, "summary（建议方案的简明概述）："),
            ("rational_advice.semantic_notation", False, IS_STR, "semantic_notation（该建议的标准化语义标识）："),  # ← 用于跨模块关联
            ("rational_advice.evidence", False, IS_LIST, "evidence（建议所依据的分析层 semantic_notation 列表）："),

            ("rational_advice.safety_first_intervention", False, IS_LIST,
             "safety_first_intervention（优先确保低位者安全的最小可行干预措施）："),
            ("rational_advice.systemic_leverage_point", False, IS_LIST,
             "systemic_leverage_point（可撬动系统动态的关键支点）："),

            # 分阶段策略（保持不变，已很完善）
            ("rational_advice.incremental_strategy", False, IS_LIST, "incremental_strategy（分阶段、低风险的行动策略）："),
            ("rational_advice.incremental_strategy.*.action", True, IS_STR, "action（具体可执行的行为动作）："),
            ("rational_advice.incremental_strategy.*.timing_or_condition", False, IS_STR,
             "timing_or_condition（执行该动作的时机或触发条件）："),
            ("rational_advice.incremental_strategy.*.required_resource", False, IS_STR,
             "required_resource（执行所需且已提及的资源）："),
            ("rational_advice.incremental_strategy.*.potential_risk", False, IS_STR, "potential_risk（可能引发的负面反应或风险）："),
            ("rational_advice.incremental_strategy.*.contingency_response", False, IS_STR,
             "contingency_response（风险发生时的应对措施）："),

            # 回退计划：改为结构化（每条含 condition + action）
            ("rational_advice.fallback_plan", False, IS_LIST,
             "fallback_plan（高风险触发时的最小安全回退措施）："),
            ("rational_advice.fallback_plan.*.trigger_condition", True, IS_STR,
             "trigger_condition（触发该回退措施的具体信号或条件）："),
            ("rational_advice.fallback_plan.*.fallback_action", True, IS_STR,
             "fallback_action（执行的最小安全行动）："),

            ("rational_advice.long_term_exit_path", False, IS_LIST,
             "long_term_exit_path（可持续脱离当前结构的现实路径）："),
            ("rational_advice.available_social_support_reinterpretation", False, IS_LIST,  # ← 修复结尾空格！
             "available_social_support_reinterpretation（对现有支持网络的重新解读与激活方式）："),

            # 利益相关方代价（保持）
            ("rational_advice.stakeholder_tradeoffs", False, IS_DICT, "stakeholder_tradeoffs（各方代价评估）："),
            ("rational_advice.stakeholder_tradeoffs.victim_cost", False, IS_LIST,
             "victim_cost（低位者可能承担的风险或损失）："),
            ("rational_advice.stakeholder_tradeoffs.oppressor_loss", False, IS_LIST,
             "oppressor_loss（高位者可能失去的资源、特权或控制力）："),
            ("rational_advice.stakeholder_tradeoffs.system_stability", False, IS_LIST,
             "system_stability（对家庭/组织短期稳定性的潜在冲击）："),
            ("rational_advice.stakeholder_tradeoffs.evidence", False, IS_LIST,
             "evidence（代价评估所依据的原文或推理）：")
        ]
    }
}

# === 基线策略：严格模式（适用于感知层、参与者提取等）===
STRICT_IRON_LAW_POLICY = {
    "context_isolation": True,
    "field_existence": "omit_if_absent",  # 可选字段未出现则省略
    "literalism": True,  # 禁止推断，必须字面依据
    "structure_consistency": True,
    "output_clean_json": True,
    "semantic_atomic": True,  # 要求 snake_case / 动词+宾语
    "max_capture_min_fabrication": True
}

# === 步骤级策略覆盖表（按 step 名称定制）===
STEP_POLICY_OVERRIDES: Dict[str, Dict] = {
    "LLM_EXPLICIT_MOTIVATION_EXTRACTION": {
        "literalism": False,  # 允许从心理描写、自由间接引语中提取
        "allow_metaphor_based_intent": True,
        "allow_rhetorical_questions": True
    },
    "LLM_INFERENCE": {
        "context_isolation": False,  # 推理层需访问感知层输出
        "literalism": False,  # 本就是推理，当然要推
        "output_clean_json": True,  # 但输出仍需干净
    },
    "LLM_RATIONAL_ADVICE": {
        "literalism": False,  # 建议需基于推理结果生成
        "context_isolation": False,
    }
    # 其他步骤如 temporal/spatial 等保持 STRICT，无需列出
}


def get_effective_policy(step_name: str) -> Dict:
    """合并基线策略与步骤特例"""
    base = STRICT_IRON_LAW_POLICY.copy()
    override = STEP_POLICY_OVERRIDES.get(step_name, {})
    base.update(override)
    return base


def render_iron_law_from_policy(policy: Dict) -> str:
    """将策略字典渲染为自然语言铁律文本"""
    lines = ["### 【其他必须遵守的通用铁律】"]

    if policy.get("context_isolation"):
        lines.append("1. 【上下文隔离原则】")
        lines.append("   - 完全无视历史对话与外部知识，仅依据当前输入块与当前指令执行。")
        lines.append("   - 禁止参考、延续或模仿任何过往输出内容。")
    else:
        lines.append("1. 【上下文感知原则】")
        lines.append("   - 可安全访问已验证的上游输出（如感知层结果）作为当前任务的合法依据。")

    if policy.get("field_existence") == "omit_if_absent":
        lines.append("2. 【存在性保守原则】")
        lines.append("   - 输出结构中的可选成分，仅当输入中存在直接、明确、字面匹配的内容时才可出现；")
        lines.append("     否则必须彻底省略（不得以 null、\"\"、[] 或字段名占位）。")
        lines.append("   - 必填成分必须存在，并按 schema 要求返回合法空值（如 \"\", [], {}）。")

    if policy.get("literalism"):
        lines.append("3. 【字面锚定原则】")
        lines.append("   - 所有输出内容必须能在输入中找到逐字或语义等价的原文片段作为唯一依据。")
        lines.append("   - 禁止任何形式的推理、补全、常识调用、角色默认、否定转肯定、功能反推或隐含归属。")
    else:
        lines.append("3. 【有限推演许可原则】")
        permitted = []
        if policy.get("allow_metaphor_based_intent"):
            permitted.append("具象化心理描写（如‘恐惧像藤蔓绞紧心脏’）中直接关联行为动因或身份认知的意图")
        if policy.get("allow_rhetorical_questions"):
            permitted.append("反问或自问句（如‘我究竟是谁？’）作为身份困惑的显式证据")
        if permitted:
            lines.append("   - 允许从以下类型的非字面表达中提取结构化语义：")
            for item in permitted:
                lines.append(f"     • {item}")
        else:
            lines.append("   - 允许基于任务目标进行必要推理，但所有结论仍需有输入中的显式语义锚点。")
        lines.append("   - 严禁无锚点的常识联想、角色补全或跨域泛化。")

    if policy.get("structure_consistency"):
        lines.append("4. 【结构一致性原则】")
        lines.append("   - 输出必须严格遵循指定 schema：字段名、类型、嵌套层级、序列格式均不可偏离。")
        lines.append("   - 数组必须为 [\"...\"] 形式，禁止裸字符串、null 或混合类型。")

    if policy.get("output_clean_json"):
        lines.append("5. 【输出洁净原则】")
        lines.append("   - 仅返回目标数据结构本身，无前缀、后缀、解释、注释、Markdown 或额外文本。")

    if policy.get("semantic_atomic"):
        lines.append("6. 【语义原子化原则】")
        lines.append("   - 所有字符串值应为标准化标签（优先 snake_case 或动词+宾语形式），")
        lines.append("     禁止完整句子、模糊形容词或文学性描述。")

    if policy.get("max_capture_min_fabrication"):
        lines.append("7. 【最大捕获最小编造原则】")
        if policy.get("literalism"):
            lines.append("   - 在严格字面约束下，穷尽所有可被直接锚定的语义单元；绝不生成无法验证的内容。")
        else:
            lines.append("   - 在当前任务允许的表达范围内（包括心理描写、自问等），")
            lines.append("     穷尽所有可结构化的显式语义；但禁止无依据扩展。")
        lines.append("   - 目标：不错过，不编造。")

    return "\n".join(lines)
