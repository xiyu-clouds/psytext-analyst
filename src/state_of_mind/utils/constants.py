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
    COMMON_SUGGESTION = "common_suggestion"
    CONSISTENCY_SUGGESTION = "consistency_suggestion"


# 预处理 并行 串行
PREPROCESSING = "preprocessing"
PARALLEL = "parallel"
SERIAL = "serial"

# 大模型预处理
LLM_SOURCE_EXTRACTION = "LLM_SOURCE_EXTRACTION"

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
LLM_DEEP_ANALYSIS = "LLM_DEEP_ANALYSIS"
LLM_RATIONAL_ADVICE = "LLM_RATIONAL_ADVICE"


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
        LLM_SOURCE_EXTRACTION: [
            ("participants", False, IS_LIST, "participants（参与者列表）："),
            ("participants.*.entity", False, IS_STR, "entity（唯一标识符）："),
            ("participants.*.name", False, IS_STR, "name（角色姓名或常用称呼）："),
            ("participants.*.social_role", False, IS_STR, "social_role（在当前情境中的静态社会角色）："),
            ("participants.*.age_range", False, IS_STR, "age_range（年龄范围）："),
            ("participants.*.gender", False, IS_STR, "gender（性别身份或表达）："),
            ("participants.*.ethnicity_or_origin", False, IS_STR, "ethnicity_or_origin（族群、国籍或地域出身）："),
            ("participants.*.physical_traits", False, IS_LIST, "physical_traits（固有生理特征，不可变或长期存在）："),
            ("participants.*.appearance", False, IS_LIST, "appearance（稳定外貌或装扮特征，视觉可辨）："),
            ("participants.*.baseline_health", False, IS_STR, "baseline_health（基础健康状况或慢性病史）："),
            ("participants.*.inherent_odor", False, IS_LIST, "inherent_odor（固有体味或气味特征）："),
            ("participants.*.voice_quality", False, IS_STR, "voice_quality（固有嗓音特质）："),
            ("participants.*.affective_orientation", False, IS_LIST, "affective_orientation（情感依恋风格）："),
            ("participants.*.personality_traits", False, IS_LIST, "personality_traits（长期人格特质）："),
            ("participants.*.behavioral_tendencies", False, IS_LIST, "behavioral_tendencies（稳定行为倾向或习惯）："),
            ("participants.*.education_level", False, IS_STR, "education_level（教育程度）："),
            ("participants.*.occupation", False, IS_STR, "occupation（职业身份）："),
            ("participants.*.socioeconomic_status", False, IS_STR, "socioeconomic_status（社会经济地位）："),
            ("participants.*.cultural_identity", False, IS_LIST, "cultural_identity（文化身份标签）："),
            ("participants.*.primary_language", False, IS_STR, "primary_language（主要使用语言）："),
        ],

        # ────────────────────────────────────────
        # 2. 时间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TEMPORAL_EXTRACTION: [
            ("temporal", False, IS_DICT, "temporal（时间感知根对象）："),
            # —— 精确时间（原文显式出现，不做归一化）
            ("temporal.exact_literals", False, IS_LIST, "exact_literals（原文中显式出现的精确时间字面量）："),
            # —— 模糊/相对时间表达
            ("temporal.relative_expressions", False, IS_LIST, "relative_expressions（原文中的相对或模糊时间表达）："),
            # —— 时间锚点（用于解析相对时间的上下文参考，如文档时间、当前时间等）
            ("temporal.reference_anchor", False, IS_STR, "reference_anchor（相对时间解析所依赖的参考时间锚点）："),
            # —— 时间范围（起止时间对，适用于“从...到...”类表达）
            ("temporal.time_ranges", False, IS_LIST, "time_ranges（原文中出现的时间区间）："),
            # —— 持续时长（如“持续两小时”、“为期三天”）
            ("temporal.durations", False, IS_LIST, "durations（原文中提及的持续时间表达）："),
            # —— 频率/周期性（如“每天”、“每周一”、“每月初”）
            ("temporal.frequencies", False, IS_LIST, "frequencies（原文中出现的周期性或频率表达）："),
            # —— 时间主体（谁经历/提及该时间）
            ("temporal.experiencer", False, IS_STR, "experiencer（时间事件的感知或陈述主体）："),
            # —— 原文证据片段（支持每个时间要素的原始文本）
            ("temporal.evidence", False, IS_LIST, "evidence（支撑时间判断的原始文本片段）："),
            # —— 事件语义标识（标准化语义标识）
            ("temporal.semantic_notation", False, IS_STR, "semantic_notation（时间事件的语义标识）："),
            # —— 客观摘要（整合上述信息，不推理、不补全）
            ("temporal.summary", False, IS_STR, "summary（基于提取内容生成的客观时间情景摘要）：")
        ],

        # ────────────────────────────────────────
        # 3. 空间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_SPATIAL_EXTRACTION: [
            ("spatial", False, IS_DICT, "spatial（空间感知根对象）："),
            ("spatial.places", False, IS_LIST, "places（原文中提及的具体地点或场所名称）："),
            ("spatial.layout_descriptions", False, IS_LIST, "layout_descriptions（原文中对空间结构或布局的描述）："),
            ("spatial.experiencer", False, IS_STR, "experiencer（空间描述的感知或陈述主体）："),
            ("spatial.proximity_relations", False, IS_LIST, "proximity_relations（空间参与者之间的关系实例列表）："),
            ("spatial.evidence", False, IS_LIST, "evidence（支撑空间判断的原始文本片段）："),
            ("spatial.semantic_notation", False, IS_STR, "semantic_notation（空间事件的标准化语义标识）："),
            ("spatial.summary", False, IS_STR, "summary（基于提取内容生成的客观空间情景摘要）："),

            ("spatial.proximity_relations.*.actor", False, IS_STR, "actor（空间关系中的主动方或参照主体）："),
            ("spatial.proximity_relations.*.target", False, IS_STR, "target（空间关系中的目标方或被参照对象）："),
            ("spatial.proximity_relations.*.distance_cm", False, IS_INT, "distance_cm（若原文明确提及，以厘米为单位的物理距离）："),
            ("spatial.proximity_relations.*.medium", False, IS_LIST, "medium（信息或互动所依赖的物理/感知媒介）："),
            ("spatial.proximity_relations.*.channel", False, IS_LIST, "channel（互动所使用的渠道或方式）："),
            ("spatial.proximity_relations.*.barrier", False, IS_LIST, "barrier（明确指出的阻碍感知或移动的障碍物）："),
            ("spatial.proximity_relations.*.relation_type", False, IS_STR,
             "relation_type（空间关系类型，）：")
        ],

        # ────────────────────────────────────────
        # 4. 视觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_VISUAL_EXTRACTION: [
            ("visual", False, IS_DICT, "visual（视觉感知根对象）："),
            ("visual.events", False, IS_LIST, "events（视觉事件列表）："),
            ("visual.evidence", False, IS_LIST, "evidence（支撑整体视觉判断的原文片段）："),
            ("visual.semantic_notation", False, IS_STR, "semantic_notation（整体视觉场景的标准化语义标识）："),
            ("visual.summary", False, IS_STR, "summary（基于客观提取生成的视觉情景摘要）："),

            ("visual.events.*.experiencer", False, IS_STR, "experiencer（观察主体）："),
            ("visual.events.*.observed_entity", False, IS_STR, "observed_entity（被观察的对象或主体）："),
            ("visual.events.*.visual_objects", False, IS_LIST, "visual_objects（原文中明确提及的可见物体）："),
            ("visual.events.*.visual_attributes", False, IS_LIST, "visual_attributes（对象的视觉属性）："),
            ("visual.events.*.visual_actions", False, IS_LIST, "visual_actions（可见的动作或姿态）："),
            ("visual.events.*.gaze_target", False, IS_STR, "gaze_target（注视目标）："),
            ("visual.events.*.eye_contact", False, IS_LIST, "eye_contact（眼神交互描述）："),
            ("visual.events.*.facial_cues", False, IS_LIST, "facial_cues（面部表情或微表情线索）："),
            ("visual.events.*.salience", False, IS_FLOAT, "salience（该视觉观察的显著性或确定性）："),
            ("visual.events.*.evidence", False, IS_LIST, "evidence（支撑该观察的原文片段）："),
            ("visual.events.*.semantic_notation", False, IS_STR, "semantic_notation（该视觉事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 5. 听觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_AUDITORY_EXTRACTION: [
            ("auditory", False, IS_DICT, "auditory（听觉感知根对象）："),
            ("auditory.events", False, IS_LIST, "events（听觉事件列表）："),
            ("auditory.evidence", False, IS_LIST, "evidence（支撑整体听觉判断的原文片段）："),
            ("auditory.semantic_notation", False, IS_STR, "semantic_notation（整体听觉场景的标准化语义标识）："),
            ("auditory.summary", False, IS_STR, "summary（基于客观提取生成的听觉情景摘要）："),

            ("auditory.events.*.experiencer", False, IS_STR, "experiencer（听觉接收主体）："),
            ("auditory.events.*.sound_source", False, IS_STR, "sound_source（发声主体或声源）："),
            ("auditory.events.*.auditory_content", False, IS_LIST, "auditory_content（直接描述的听觉内容关键词或原文片段）："),
            ("auditory.events.*.is_primary_focus", False, IS_BOOL, "is_primary_focus（是否为当前听觉焦点）："),
            ("auditory.events.*.rhetorical_patterns", False, IS_LIST, "rhetorical_patterns（直接使用的修辞结构或术语）："),
            ("auditory.events.*.prosody_cues", False, IS_LIST, "prosody_cues（直接描述的声音特征）："),
            ("auditory.events.*.pause_description", False, IS_STR, "pause_description（明确描述的停顿特征）："),
            ("auditory.events.*.intensity", False, IS_FLOAT, "intensity（听觉感知强度，基于修饰词量化）："),
            ("auditory.events.*.evidence", False, IS_LIST, "evidence（支撑该听觉事件的原文片段）："),
            ("auditory.events.*.semantic_notation", False, IS_STR, "semantic_notation（该听觉事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 6. 嗅觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_OLFACTORY_EXTRACTION: [
            ("olfactory", False, IS_DICT, "olfactory（嗅觉感知根对象）："),
            ("olfactory.events", False, IS_LIST, "events（嗅觉事件列表）："),
            ("olfactory.evidence", False, IS_LIST, "evidence（支撑整体嗅觉判断的原文片段）："),
            ("olfactory.semantic_notation", False, IS_STR, "semantic_notation（整体嗅觉场景的标准化语义标识）："),
            ("olfactory.summary", False, IS_STR, "summary（基于客观提取生成的嗅觉情景摘要）："),

            ("olfactory.events.*.experiencer", False, IS_STR, "experiencer（气味感知主体）："),
            ("olfactory.events.*.odor_source", False, IS_STR, "odor_source（气味来源主体或对象）："),
            ("olfactory.events.*.odor_descriptors", False, IS_LIST, "odor_descriptors（直接出现的气味描述词或短语）："),
            ("olfactory.events.*.intensity", False, IS_FLOAT, "intensity（嗅觉感知强度，基于修饰词量化）："),
            ("olfactory.events.*.evidence", False, IS_LIST, "evidence（支撑该嗅觉事件的原文片段）："),
            ("olfactory.events.*.semantic_notation", False, IS_STR, "semantic_notation（该嗅觉事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 7. 触觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TACTILE_EXTRACTION: [
            ("tactile", False, IS_DICT, "tactile（触觉感知根对象）："),
            ("tactile.events", False, IS_LIST, "events（触觉事件列表）："),
            ("tactile.evidence", False, IS_LIST, "evidence（支撑整体触觉判断的原文片段）："),
            ("tactile.semantic_notation", False, IS_STR, "semantic_notation（整体触觉场景的标准化语义标识）："),
            ("tactile.summary", False, IS_STR, "summary（基于客观提取生成的触觉情景摘要）："),

            ("tactile.events.*.experiencer", False, IS_STR, "experiencer（触觉体验主体）："),
            ("tactile.events.*.contact_target", False, IS_STR, "contact_target（被接触对象或身体部位）："),
            ("tactile.events.*.tactile_descriptors", False, IS_LIST, "tactile_descriptors（直接描述的触觉感受或动作）："),
            ("tactile.events.*.intensity", False, IS_FLOAT, "intensity（触觉感知强度，基于修饰词量化）："),
            ("tactile.events.*.evidence", False, IS_LIST, "evidence（支撑该触觉事件的原文片段）："),
            ("tactile.events.*.semantic_notation", False, IS_STR, "semantic_notation（该触觉事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 8. 味觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_GUSTATORY_EXTRACTION: [
            ("gustatory", False, IS_DICT, "gustatory（味觉感知根对象）："),
            ("gustatory.events", False, IS_LIST, "events（味觉事件列表）："),
            ("gustatory.evidence", False, IS_LIST, "evidence（支撑整体味觉判断的原文片段）："),
            ("gustatory.semantic_notation", False, IS_STR, "semantic_notation（整体味觉场景的标准化语义标识）："),
            ("gustatory.summary", False, IS_STR, "summary（基于客观提取生成的味觉情景摘要）："),

            ("gustatory.events.*.experiencer", False, IS_STR, "experiencer（味觉体验主体）："),
            ("gustatory.events.*.taste_source", False, IS_STR, "taste_source（食物或味道来源）："),
            ("gustatory.events.*.taste_descriptors", False, IS_LIST, "taste_descriptors（直接描述的味道或短语）："),
            ("gustatory.events.*.intensity", False, IS_FLOAT, "intensity（味觉感知强度，基于修饰词量化）："),
            ("gustatory.events.*.evidence", False, IS_LIST, "evidence（支撑该味觉事件的原文片段）："),
            ("gustatory.events.*.semantic_notation", False, IS_STR, "semantic_notation（该味觉事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 9. 内感受
        # ────────────────────────────────────────
        LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION: [
            ("interoceptive", False, IS_DICT, "interoceptive（内感受感知根对象）："),
            ("interoceptive.events", False, IS_LIST, "events（内感受事件列表）："),
            ("interoceptive.evidence", False, IS_LIST, "evidence（支撑整体内感受判断的原文片段）："),
            ("interoceptive.semantic_notation", False, IS_STR, "semantic_notation（整体内感受场景的标准化语义标识）："),
            ("interoceptive.summary", False, IS_STR, "summary（基于客观提取生成的内感受情景摘要）："),

            ("interoceptive.events.*.experiencer", False, IS_STR, "experiencer（主观感受的体验者）："),
            ("interoceptive.events.*.body_sensation", False, IS_LIST, "body_sensation（直接描述的身体内部感觉）："),
            ("interoceptive.events.*.intensity", False, IS_FLOAT, "intensity（内感受强度，基于修饰词量化）："),
            ("interoceptive.events.*.evidence", False, IS_LIST, "evidence（支撑该内感受事件的原文片段）："),
            ("interoceptive.events.*.semantic_notation", False, IS_STR, "semantic_notation（该内感受事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 10. 认知过程
        # ────────────────────────────────────────
        LLM_PERCEPTION_COGNITIVE_EXTRACTION: [
            ("cognitive", False, IS_DICT, "cognitive（认知过程根对象）："),
            ("cognitive.events", False, IS_LIST, "events（认知事件列表）："),
            ("cognitive.evidence", False, IS_LIST, "evidence（支撑整体认知判断的原文片段）："),
            ("cognitive.semantic_notation", False, IS_STR, "semantic_notation（整体认知场景的标准化语义标识）："),
            ("cognitive.summary", False, IS_STR, "summary（基于客观提取生成的认知情景摘要）："),

            ("cognitive.events.*.experiencer", False, IS_STR, "experiencer（认知主体）："),
            ("cognitive.events.*.explicit_thought", False, IS_LIST, "explicit_thought（直接表达的思维内容）："),
            ("cognitive.events.*.intensity", False, IS_FLOAT, "intensity（认知负荷强度）："),
            ("cognitive.events.*.evidence", False, IS_LIST, "evidence（支撑该认知事件的原文片段）："),
            ("cognitive.events.*.semantic_notation", False, IS_STR, "semantic_notation（该认知事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 11. 躯体化表现
        # ────────────────────────────────────────
        LLM_PERCEPTION_BODILY_EXTRACTION: [
            ("bodily", False, IS_DICT, "bodily（躯体化表现根对象）："),
            ("bodily.events", False, IS_LIST, "events（躯体化事件列表）："),
            ("bodily.evidence", False, IS_LIST, "evidence（支撑整体躯体化判断的原文片段）："),
            ("bodily.semantic_notation", False, IS_STR, "semantic_notation（整体躯体化场景的标准化语义标识）："),
            ("bodily.summary", False, IS_STR, "summary（基于客观提取生成的躯体化情景摘要）："),

            ("bodily.events.*.experiencer", False, IS_STR, "experiencer（躯体行为主体）："),
            ("bodily.events.*.observable_behavior", False, IS_LIST, "observable_behavior（直接描述的外部可观测身体行为）："),
            ("bodily.events.*.intensity", False, IS_FLOAT, "intensity（躯体化表现症状强度）："),
            ("bodily.events.*.evidence", False, IS_LIST, "evidence（支撑该躯体化事件的原文片段）："),
            ("bodily.events.*.semantic_notation", False, IS_STR, "semantic_notation（该躯体化事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 12. 情感状态
        # ────────────────────────────────────────
        LLM_PERCEPTION_EMOTIONAL_EXTRACTION: [
            ("emotional", False, IS_DICT, "emotional（情感状态根对象）："),
            ("emotional.events", False, IS_LIST, "events（情感事件列表）："),
            ("emotional.evidence", False, IS_LIST, "evidence（支撑整体情感判断的原文片段）："),
            ("emotional.semantic_notation", False, IS_STR, "semantic_notation（整体情感场景的标准化语义标识）："),
            ("emotional.summary", False, IS_STR, "summary（基于客观提取生成的情感情景摘要）："),

            ("emotional.events.*.experiencer", False, IS_STR, "experiencer（情绪表达主体）："),
            ("emotional.events.*.emotion_labels", False, IS_LIST, "emotion_labels（具体情绪标签）："),
            ("emotional.events.*.valence", False, IS_FLOAT, "valence（情绪效价）："),
            ("emotional.events.*.arousal", False, IS_FLOAT, "arousal（情绪唤醒度）："),
            ("emotional.events.*.intensity", False, IS_FLOAT, "intensity（情感强度）："),
            ("emotional.events.*.evidence", False, IS_LIST, "evidence（支撑该情感判断的原文片段）："),
            ("emotional.events.*.semantic_notation", False, IS_STR, "semantic_notation（该情感事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 13. 社会关系
        # ────────────────────────────────────────
        LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION: [
            ("social_relation", False, IS_DICT, "social_relation（社会关系根对象）："),
            ("social_relation.events", False, IS_LIST, "events（社会关系事件列表）："),
            ("social_relation.evidence", False, IS_LIST, "evidence（支撑整体关系判断的原文片段）："),
            ("social_relation.semantic_notation", False, IS_STR, "semantic_notation（整体社会关系场景的标准化语义标识）："),
            ("social_relation.summary", False, IS_STR, "summary（基于客观提取生成的社会关系情景摘要）："),

            ("social_relation.events.*.experiencer", False, IS_LIST, "participants（关系涉及的参与者）："),
            ("social_relation.events.*.relation_type", False, IS_LIST, "relation_type（直接提取的关系类型关键词）："),
            ("social_relation.events.*.explicit_relation_statement", False, IS_LIST,
             "explicit_relation_statement（直接陈述的关系信息）："),
            ("social_relation.events.*.evidence", False, IS_LIST, "evidence（支撑该关系事件的原文片段）："),
            ("social_relation.events.*.semantic_notation", False, IS_STR, "semantic_notation（该社会关系事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 14. 推理层
        # ────────────────────────────────────────
        LLM_INFERENCE: [
            ("inference", False, IS_DICT, "inference（推理层根对象）："),
            ("inference.events", False, IS_LIST, "events（推理事件列表）："),
            ("inference.evidence", False, IS_LIST, "evidence（支撑整体推理的原始文本依据）："),
            ("inference.semantic_notation", False, IS_STR, "semantic_notation（整体推理场景的标准化语义标识）："),
            ("inference.summary", False, IS_STR, "summary（基于锚点事件推导出的情景性结论摘要）："),

            ("inference.events.*.experiencer", False, IS_STR, "experiencer（基于谁推理的主体）："),
            ("inference.events.*.inference_type", False, IS_STR, "inference_type（推理类型）："),
            ("inference.events.*.anchor_points", False, IS_LIST, "anchor_points（所依赖的感知层事件 semantic_notation 列表）："),
            ("inference.events.*.inferred_proposition", False, IS_STR, "inferred_proposition（用一句话陈述的推理结论）："),
            ("inference.events.*.evidence", False, IS_LIST, "evidence（支撑该推理的原始文本片段）："),
            ("inference.events.*.semantic_notation", False, IS_STR, "semantic_notation（该推理事件的标准化语义标识）：")
        ],

        # ────────────────────────────────────────
        # 15. 深度分析
        # ────────────────────────────────────────
        LLM_DEEP_ANALYSIS: [
            ("deep_analysis", False, IS_DICT, "deep_analysis（深度分析根对象）："),
            ("deep_analysis.events", False, IS_LIST, "events（深度分析事件列表）："),
            ("deep_analysis.evidence", False, IS_LIST, "evidence（支撑整体分析的原始文本依据）："),
            ("deep_analysis.semantic_notation", False, IS_STR, "semantic_notation（整体深度分析场景的标准化语义标识）："),
            ("deep_analysis.summary", False, IS_STR, "summary（基于显性陈述整合的深层动因摘要）："),

            ("deep_analysis.events.*.experiencer", False, IS_STR, "experiencer（深度分析的参与者主体对象）："),
            # 核心驱动力（用户亲口说的“我之所以...是因为...”）
            ("deep_analysis.events.*.core_driver", False, IS_LIST, "core_driver（用户明确表达的根本需求、恐惧或动机）："),
            # 关怀与担忧动机结构 <<<
            ("deep_analysis.events.*.care_expression", False, IS_LIST, "care_expression（明确表达的关怀行为或意图）："),
            ("deep_analysis.events.*.separation_anxiety", False, IS_LIST, "separation_anxiety（因分离而显式陈述的担忧、恐惧或回忆）："),
            ("deep_analysis.events.*.protective_intent", False, IS_LIST, "protective_intent（为对方健康、安全或福祉采取行动的直接表述）："),
            # 权力不对称（仅当原文提及控制、依赖、威胁时提取）
            ("deep_analysis.events.*.power_asymmetry", False, IS_DICT, "power_asymmetry（权力差异结构）："),
            ("deep_analysis.events.*.power_asymmetry.control_axis", False, IS_LIST, "control_axis（明确提到的控制维度）："),
            ("deep_analysis.events.*.power_asymmetry.dependency_ratio", False, IS_FLOAT,
             "dependency_ratio（依赖程度，仅当有量化表述如“完全靠他”时赋值）："),
            ("deep_analysis.events.*.power_asymmetry.threat_vector", False, IS_LIST, "threat_vector（直接陈述的威胁方式）："),
            ("deep_analysis.events.*.power_asymmetry.evidence", False, IS_LIST, "evidence（支撑权力分析的原文片段）："),
            # 资源控制
            ("deep_analysis.events.*.resource_control", False, IS_LIST, "resource_control（明确指出对方掌控的关键资源）："),
            # 生存性服从
            ("deep_analysis.events.*.survival_imperative", False, IS_LIST, "survival_imperative（亲口表达的服从理由）："),
            # 社会规范压力
            ("deep_analysis.events.*.social_enforcement_mechanism", False, IS_LIST,
             "social_enforcement_mechanism（提及的社会规范、家庭压力或群体期待）："),
            # 话术分析（仅提取原话）
            ("deep_analysis.events.*.narrative_distortion", False, IS_DICT, "narrative_distortion（话术策略）："),
            ("deep_analysis.events.*.narrative_distortion.self_justification", False, IS_STR, "self_justification（为自身行为提供的直接合理化语句）："),
            ("deep_analysis.events.*.narrative_distortion.blame_shift", False, IS_STR, "blame_shift（明确转嫁责任的原话）："),
            ("deep_analysis.events.*.narrative_distortion.moral_licensing", False, IS_STR, "moral_licensing（以道德身份豁免行为的原话）："),
            ("deep_analysis.events.*.narrative_distortion.evidence", False, IS_LIST, "evidence（支撑话术分析的原文片段）：")
        ],

        # ────────────────────────────────────────
        # 16. 合理建议
        # ────────────────────────────────────────
        LLM_RATIONAL_ADVICE: [
            ("rational_advice", False, IS_DICT, "rational_advice（合理建议根对象）："),
            ("rational_advice.evidence", False, IS_LIST, "evidence（建议所依据的分析层 semantic_notation 列表）："),
            ("rational_advice.semantic_notation", False, IS_STR, "semantic_notation（整体建议方案的标准化语义标识）："),
            ("rational_advice.summary", False, IS_STR, "summary（建议方案的简明概述）："),
            # 安全优先干预
            ("rational_advice.safety_first_intervention", False, IS_LIST, "safety_first_intervention（优先确保低位者安全的最小可行干预措施）："),
            # 系统杠杆点
            ("rational_advice.systemic_leverage_point", False, IS_LIST, "systemic_leverage_point（可撬动系统动态的关键支点）："),
            # 分阶段策略
            ("rational_advice.incremental_strategy", False, IS_LIST, "incremental_strategy（分阶段、低风险的行动策略）："),
            # 利益相关方代价（结构化）
            ("rational_advice.stakeholder_tradeoffs", False, IS_DICT, "stakeholder_tradeoffs（各方代价评估）："),
            ("rational_advice.stakeholder_tradeoffs.victim_cost", False, IS_LIST, "victim_cost（低位者可能承担的风险或损失）："),
            ("rational_advice.stakeholder_tradeoffs.oppressor_loss", False, IS_LIST, "oppressor_loss（高位者可能失去的资源、特权或控制力）："),
            ("rational_advice.stakeholder_tradeoffs.system_stability", False, IS_LIST, "system_stability（对家庭/组织短期稳定性的潜在冲击）："),
            ("rational_advice.stakeholder_tradeoffs.evidence", False, IS_LIST, "evidence（代价评估的依据）："),
            # 长期脱离路径
            ("rational_advice.long_term_exit_path", False, IS_LIST, "long_term_exit_path（可持续脱离当前结构的现实路径）："),
            # 文化适应
            ("rational_advice.cultural_adaptation_needed", False, IS_LIST, "cultural_adaptation_needed（需调整的文化认知或可寻求的社会支持）：")
        ]
    }
}
