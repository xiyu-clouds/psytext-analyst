from typing import Dict, List
from src.state_of_mind.types.perception import ValidationRule
from src.state_of_mind.utils.data_validator import IS_DICT, IS_STR, IS_LIST, IS_FLOAT, IS_INT, IS_BOOL

# 模板类别
CATEGORY_RAW = "raw"
CATEGORY_SUGGESTION = "suggestion"
COREFERENCE_RESOLUTION_BATCH = "coreference_resolution_batch"

# 预处理 并行 串行
PREPROCESSING = "preprocessing"
PARALLEL = "parallel"
SERIAL = "serial"

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
            ("participants.*.family_status ", False, IS_STR, "family_status （家庭状态标签）："),
            ("participants.*.education_level", False, IS_STR, "education_level（教育程度）："),
            ("participants.*.cultural_identity", False, IS_LIST, "cultural_identity（文化/民族/地域身份标签）："),
            ("participants.*.primary_language", False, IS_STR, "primary_language（主要使用语言）："),
            ("participants.*.institutional_affiliation", False, IS_LIST, "institutional_affiliation（所属机构标签）："),

            # === 生物与生理属性 ===
            ("participants.*.age_range", False, IS_STR, "age_range（年龄范围）："),
            ("participants.*.gender", False, IS_STR, "gender（性别或相关表述）："),
            ("participants.*.ethnicity_or_origin", False, IS_STR, "ethnicity_or_origin（族群/国籍/地域出身）："),
            ("participants.*.physical_traits", False, IS_LIST, "physical_traits（固有生理特征）："),
            ("participants.*.current_physical_state", False, IS_STR, "current_physical_state（当前身体状态）："),
            ("participants.*.visible_injury_or_wound", False, IS_LIST, "visible_injury_or_wound（可见伤痕或包扎）："),

            # === 感官与外显特征 ===
            ("participants.*.appearance", False, IS_LIST, "appearance（外貌与衣着特征）："),
            ("participants.*.voice_quality", False, IS_STR, "voice_quality（嗓音特质）："),
            ("participants.*.inherent_odor", False, IS_LIST, "inherent_odor（体味或气息）："),
            ("participants.*.carried_objects", False, IS_LIST, "carried_objects（持有物品）："),
            ("participants.*.worn_technology", False, IS_LIST, "worn_technology（佩戴的电子设备）："),

            # === 心理与行为属性 ===
            ("participants.*.personality_traits", False, IS_LIST, "personality_traits（性格特质词汇）："),
            ("participants.*.behavioral_tendencies", False, IS_LIST, "behavioral_tendencies（行为习惯表述）："),
            ("participants.*.speech_pattern", False, IS_STR, "speech_pattern（说话方式）："),

            # === 交互与场景角色 ===
            ("participants.*.interaction_role", False, IS_STR, "interaction_role（当前互动中的临时角色）："),
        ],

        # ────────────────────────────────────────
        # 动态筛选存在哪些感知维度信息
        # ────────────────────────────────────────
        LLM_DIMENSION_GATE:[
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
        # 提前判断是否值得开启高阶三步链
        # ────────────────────────────────────────
        LLM_INFERENCE_ELIGIBILITY:[
            ("eligibility", True, IS_DICT, "eligibility（高阶步骤执行资格）："),
            ("eligibility.eligible", True, IS_BOOL, "eligible（符合条件）："),
        ],

        # ────────────────────────────────────────
        # 2. 时间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TEMPORAL_EXTRACTION: [
            ("temporal", False, IS_DICT, "temporal（时间信息根对象）："),
            ("temporal.events", False, IS_LIST, "events（时间事件列表）："),
            ("temporal.evidence", False, IS_LIST, "evidence（全局时间证据片段）："),
            ("temporal.summary", False, IS_STR, "summary（时间事件摘要）："),

            ("temporal.events.*.experiencer", False, IS_STR, "experiencer（事件陈述主体）："),
            ("temporal.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("temporal.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化时间语义标识）："),
            ("temporal.events.*.exact_literals", False, IS_LIST, "exact_literals（精确时间字面量）："),
            ("temporal.events.*.relative_expressions", False, IS_LIST, "relative_expressions（相对时间表达）："),
            ("temporal.events.*.negated_time", False, IS_LIST, "negated_time（否定句中的时间成分）："),
            ("temporal.events.*.time_ranges", False, IS_LIST, "time_ranges（时间区间表达）："),
            ("temporal.events.*.durations", False, IS_LIST, "durations（持续时间表达）："),
            ("temporal.events.*.frequencies", False, IS_LIST, "frequencies（频率或周期表达）："),
            ("temporal.events.*.event_markers", False, IS_LIST, "event_markers（共现动词或名词）："),
            ("temporal.events.*.tense_aspect", False, IS_STR, "tense_aspect（时体标记）："),
            ("temporal.events.*.seasonal_or_cultural_time", False, IS_LIST, "seasonal_or_cultural_time（制度性文化时间标识）："),
            ("temporal.events.*.temporal_anchor", False, IS_STR, "temporal_anchor（时间参考锚点）："),
            ("temporal.events.*.uncertainty_modifiers", False, IS_LIST, "uncertainty_modifiers（时间不确定性修饰语）：")
        ],

        # ────────────────────────────────────────
        # 3. 空间感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_SPATIAL_EXTRACTION: [
            ("spatial", False, IS_DICT, "spatial（空间信息根对象）："),
            ("spatial.events", False, IS_LIST, "events（空间事件列表）："),
            ("spatial.evidence", False, IS_LIST, "evidence（全局空间证据片段）："),
            ("spatial.summary", False, IS_STR, "summary（空间事件摘要）："),

            ("spatial.events.*.experiencer", False, IS_STR, "experiencer（空间陈述主体）："),
            ("spatial.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("spatial.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化空间语义标识）："),
            ("spatial.events.*.places", False, IS_LIST, "places（具体地点名称）："),
            ("spatial.events.*.layout_descriptions", False, IS_LIST, "layout_descriptions（空间布局描述）："),
            ("spatial.events.*.negated_places", False, IS_LIST, "negated_places（否定句中的地点成分）："),
            ("spatial.events.*.spatial_event_markers", False, IS_LIST, "spatial_event_markers（共现动词或名词）："),
            ("spatial.events.*.cultural_or_institutional_spaces", False, IS_LIST,
             "cultural_or_institutional_spaces（制度性或文化性空间）："),
            ("spatial.events.*.orientation_descriptions", False, IS_LIST, "orientation_descriptions（方向或朝向描述）："),

            ("spatial.events.*.proximity_relations", False, IS_LIST, "proximity_relations（显式空间关系）："),
            ("spatial.events.*.proximity_relations.*.actor", False, IS_STR, "actor（关系参照主体）："),
            ("spatial.events.*.proximity_relations.*.target", False, IS_STR, "target（关系参照对象）："),
            ("spatial.events.*.proximity_relations.*.distance_cm", False, IS_INT, "distance_cm（物理距离，单位：厘米）："),
            ("spatial.events.*.proximity_relations.*.modifiers", False, IS_LIST, "modifiers（修饰性空间成分）："),
            ("spatial.events.*.proximity_relations.*.relation_type", False, IS_STR, "relation_type（原文空间关系短语）：")
        ],

        # ────────────────────────────────────────
        # 4. 视觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_VISUAL_EXTRACTION: [
            ("visual", False, IS_DICT, "visual（视觉信息根对象）："),
            ("visual.events", False, IS_LIST, "events（视觉事件列表）："),
            ("visual.evidence", False, IS_LIST, "evidence（全局视觉证据片段）："),
            ("visual.summary", False, IS_STR, "summary（视觉事件摘要）："),

            ("visual.events.*.experiencer", False, IS_STR, "experiencer（视觉陈述主体）："),
            ("visual.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("visual.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化视觉语义标识）："),
            ("visual.events.*.observed_entity", False, IS_STR, "observed_entity（被观察对象）："),
            ("visual.events.*.visual_objects", False, IS_LIST, "visual_objects（可见物体）："),
            ("visual.events.*.visual_attributes", False, IS_LIST, "visual_attributes（视觉属性）："),
            ("visual.events.*.visual_actions", False, IS_LIST, "visual_actions（可见的动作或姿态）："),
            ("visual.events.*.gaze_target", False, IS_STR, "gaze_target（注视目标）："),
            ("visual.events.*.eye_contact", False, IS_LIST, "eye_contact（眼神交互描述）："),
            ("visual.events.*.facial_cues", False, IS_LIST, "facial_cues（面部表情描述）："),
            ("visual.events.*.salience", False, IS_FLOAT, "salience（观察确定性修饰词）："),
            ("visual.events.*.negated_observations", False, IS_LIST, "negated_observations（否定句中的观察对象）："),
            ("visual.events.*.visual_medium", False, IS_LIST, "visual_medium（视觉媒介）："),
            ("visual.events.*.occlusion_or_obstruction", False, IS_LIST, "occlusion_or_obstruction（遮挡物）："),
            ("visual.events.*.lighting_conditions", False, IS_LIST, "lighting_conditions（光照条件）："),
        ],

        # ────────────────────────────────────────
        # 5. 听觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_AUDITORY_EXTRACTION: [
            ("auditory", False, IS_DICT, "auditory（听觉信息根对象）："),
            ("auditory.events", False, IS_LIST, "events（听觉事件列表）："),
            ("auditory.evidence", False, IS_LIST, "evidence（全局听觉证据片段）："),
            ("auditory.summary", False, IS_STR, "summary（听觉事件摘要）："),

            ("auditory.events.*.experiencer", False, IS_STR, "experiencer（听觉陈述主体）："),
            ("auditory.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("auditory.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化听觉语义标识）："),
            ("auditory.events.*.sound_source", False, IS_STR, "sound_source（声源）："),
            ("auditory.events.*.auditory_content", False, IS_LIST, "auditory_content（言语内容）："),
            ("auditory.events.*.is_primary_focus", False, IS_BOOL, "is_primary_focus（是否被强调为唯一/主要听觉内容）："),
            ("auditory.events.*.prosody_cues", False, IS_LIST, "prosody_cues（语调与发声特征）："),
            ("auditory.events.*.pause_description", False, IS_STR, "pause_description（言语停顿描述）："),
            ("auditory.events.*.intensity", False, IS_FLOAT, "intensity（声音强度修饰词）："),
            ("auditory.events.*.negated_observations", False, IS_LIST, "negated_observations（否定句中的听觉对象）："),
            ("auditory.events.*.auditory_medium", False, IS_LIST, "auditory_medium（听觉媒介）："),
            ("auditory.events.*.background_sounds", False, IS_LIST, "background_sounds（背景声音）："),
            ("auditory.events.*.nonverbal_sounds", False, IS_LIST, "nonverbal_sounds（非语言声音）：")
        ],

        # ────────────────────────────────────────
        # 6. 嗅觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_OLFACTORY_EXTRACTION: [
            ("olfactory", False, IS_DICT, "olfactory（嗅觉信息根对象）："),
            ("olfactory.events", False, IS_LIST, "events（嗅觉事件列表）："),
            ("olfactory.evidence", False, IS_LIST, "evidence（全局嗅觉证据片段）："),
            ("olfactory.summary", False, IS_STR, "summary（嗅觉事件摘要）："),

            ("olfactory.events.*.experiencer", False, IS_STR, "experiencer（嗅觉陈述主体）："),
            ("olfactory.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("olfactory.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化嗅觉语义标识）："),
            ("olfactory.events.*.odor_source", False, IS_STR, "odor_source（气味来源）："),
            ("olfactory.events.*.odor_descriptors", False, IS_LIST, "odor_descriptors（气味描述词）："),
            ("olfactory.events.*.intensity", False, IS_FLOAT, "intensity（气味强度程度）："),
            ("olfactory.events.*.negated_observations", False, IS_LIST, "negated_observations（否定句中的气味对象）："),
            ("olfactory.events.*.odor_valence", False, IS_LIST, "odor_valence（气味评价词）："),
            ("olfactory.events.*.odor_source_category", False, IS_LIST, "odor_source_category（来源类别词）："),
            ("olfactory.events.*.olfactory_actions", False, IS_LIST, "olfactory_actions（嗅觉相关动作）：")
        ],

        # ────────────────────────────────────────
        # 7. 触觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_TACTILE_EXTRACTION: [
            ("tactile", False, IS_DICT, "tactile（触觉信息根对象）："),
            ("tactile.events", False, IS_LIST, "events（触觉事件列表）："),
            ("tactile.evidence", False, IS_LIST, "evidence（全局触觉证据片段）："),
            ("tactile.summary", False, IS_STR, "summary（触觉事件摘要）："),

            ("tactile.events.*.experiencer", False, IS_STR, "experiencer（触觉陈述主体）："),
            ("tactile.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("tactile.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化触觉语义标识）："),
            ("tactile.events.*.contact_target", False, IS_STR, "contact_target（接触对象或表面）："),
            ("tactile.events.*.tactile_descriptors", False, IS_LIST, "tactile_descriptors（触觉感受或动作描述）："),
            ("tactile.events.*.intensity", False, IS_FLOAT, "intensity（触觉强度程度）："),
            ("tactile.events.*.contact_initiator", False, IS_STR, "contact_initiator（接触发起方）："),
            ("tactile.events.*.body_part", False, IS_LIST, "body_part（触觉发生的身体部位）："),
            ("tactile.events.*.texture", False, IS_LIST, "texture（质地类描述）："),
            ("tactile.events.*.temperature", False, IS_LIST, "temperature（温度类描述）："),
            ("tactile.events.*.pressure", False, IS_LIST, "pressure（压力类描述）："),
            ("tactile.events.*.pain", False, IS_LIST, "pain（疼痛类描述）："),
            ("tactile.events.*.motion", False, IS_LIST, "motion（动态触觉描述）："),
            ("tactile.events.*.vibration", False, IS_LIST, "vibration（震动类描述）："),
            ("tactile.events.*.moisture", False, IS_LIST, "moisture（湿度/干湿类描述）："),
            ("tactile.events.*.contact", False, IS_LIST, "contact（接触存在性或方式描述）："),
            ("tactile.events.*.negated_observations", False, IS_LIST, "negated_observations（被否定的触觉内容）："),
            ("tactile.events.*.tactile_intent_or_valence", False, IS_LIST, "tactile_intent_or_valence（触觉情感或意图词）：")
        ],

        # ────────────────────────────────────────
        # 8. 味觉感知
        # ────────────────────────────────────────
        LLM_PERCEPTION_GUSTATORY_EXTRACTION: [
            ("gustatory", False, IS_DICT, "gustatory（味觉信息根对象）："),
            ("gustatory.events", False, IS_LIST, "events（味觉事件列表）："),
            ("gustatory.evidence", False, IS_LIST, "evidence（全局味觉证据片段）："),
            ("gustatory.summary", False, IS_STR, "summary（味觉事件摘要）："),

            ("gustatory.events.*.experiencer", False, IS_STR, "experiencer（味觉陈述主体）："),
            ("gustatory.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("gustatory.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化味觉语义标识）："),
            ("gustatory.events.*.intensity", False, IS_FLOAT, "intensity（味觉强度程度）："),
            ("gustatory.events.*.taste_source", False, IS_STR, "taste_source（味道来源或物质）："),
            ("gustatory.events.*.taste_descriptors", False, IS_LIST, "taste_descriptors（味觉感受或描述短语）："),
            ("gustatory.events.*.contact_initiator", False, IS_STR, "contact_initiator（摄入发起方）："),
            ("gustatory.events.*.body_part", False, IS_STR, "body_part（味觉发生的身体部位）："),
            ("gustatory.events.*.intent_or_valence", False, IS_LIST, "intent_or_valence（味觉情感或意图词）："),
            ("gustatory.events.*.negated_observations", False, IS_LIST, "negated_observations（被否定的味觉内容）："),
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
            ("gustatory.events.*.thermal", False, IS_LIST, "thermal（冷热感描述）：")
        ],

        # ────────────────────────────────────────
        # 9. 内感受
        # ────────────────────────────────────────
        LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION: [
            ("interoceptive", False, IS_DICT, "interoceptive（内感受信息根对象）："),
            ("interoceptive.events", False, IS_LIST, "events（内感受事件列表）："),
            ("interoceptive.evidence", False, IS_LIST, "evidence（全局内感受证据片段）："),
            ("interoceptive.summary", False, IS_STR, "summary（内感受事件摘要）："),

            ("interoceptive.events.*.experiencer", False, IS_STR, "experiencer（内感受陈述主体）："),
            ("interoceptive.events.*.intensity", False, IS_FLOAT, "intensity（内感受强度程度）："),
            ("interoceptive.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("interoceptive.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化内感受语义标识）："),
            ("interoceptive.events.*.contact_initiator", False, IS_STR, "contact_initiator（内感受触发者）："),
            ("interoceptive.events.*.body_part", False, IS_STR, "body_part（内感受发生的身体部位）："),
            ("interoceptive.events.*.intent_or_valence", False, IS_LIST,
             "tactile_intent_or_valence（内感受的情感或意图词）："),
            ("interoceptive.events.*.negated_observations", False, IS_LIST, "negated_observations（被否定的内感受内容）："),
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
            ("cognitive", False, IS_DICT, "cognitive（认知过程信息根对象）："),
            ("cognitive.events", False, IS_LIST, "events（认知事件列表）："),
            ("cognitive.evidence", False, IS_LIST, "evidence（全局认知过程证据片段）："),
            ("cognitive.summary", False, IS_STR, "summary（认知过程事件摘要）："),

            ("cognitive.events.*.experiencer", False, IS_STR, "experiencer（认知过程陈述主体）："),
            ("cognitive.events.*.intensity", False, IS_FLOAT, "intensity（认知负荷或确信强度）："),
            ("cognitive.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("cognitive.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化认知过程语义标识）："),
            ("cognitive.events.*.cognitive_agent", False, IS_STR, "cognitive_agent（思维发起者）："),
            ("cognitive.events.*.target_entity", False, IS_STR, "target_entity（思维指向的对象或主题）："),
            ("cognitive.events.*.cognitive_valence", False, IS_LIST, "cognitive_valence（认知情感倾向词）："),
            ("cognitive.events.*.negated_cognitions", False, IS_LIST, "negated_cognitions（被否定的认知过程内容）："),
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
            ("bodily", False, IS_DICT, "bodily（躯体化表现信息根对象）："),
            ("bodily.events", False, IS_LIST, "events（躯体化事件列表）："),
            ("bodily.evidence", False, IS_LIST, "evidence（全局躯体化表现证据片段）："),
            ("bodily.summary", False, IS_STR, "summary（躯体化表现事件摘要）："),

            ("bodily.events.*.experiencer", False, IS_STR, "experiencer（躯体化行为执行主体）："),
            ("bodily.events.*.intensity", False, IS_FLOAT, "intensity（躯体化表现强度）："),
            ("bodily.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("bodily.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化躯体化表现语义标识）："),
            ("bodily.events.*.observer", False, IS_STR, "observer（行为观察者）："),
            ("bodily.events.*.movement_direction", False, IS_STR, "movement_direction（运动方向/趋势）："),
            ("bodily.events.*.posture", False, IS_STR, "posture（静态身体姿态）："),
            ("bodily.events.*.facial_expression", False, IS_LIST, "facial_expression（面部表情显式描述）："),
            ("bodily.events.*.vocal_behavior", False, IS_LIST, "vocal_behavior（声音物理表现）："),
            ("bodily.events.*.autonomic_signs", False, IS_LIST, "autonomic_signs（自主神经外显征象）："),
            ("bodily.events.*.motor_behavior", False, IS_LIST, "motor_behavior（随意运动行为）："),
            ("bodily.events.*.freeze_or_faint", False, IS_LIST, "freeze_or_faint（冻结或晕厥类反应）：")
        ],

        # ────────────────────────────────────────
        # 12. 情感状态
        # ────────────────────────────────────────
        LLM_PERCEPTION_EMOTIONAL_EXTRACTION: [
            ("emotional", False, IS_DICT, "emotional（情感状态信息根对象）："),
            ("emotional.events", False, IS_LIST, "events（情感事件列表）："),
            ("emotional.evidence", False, IS_LIST, "evidence（全局情感状态证据片段）："),
            ("emotional.summary", False, IS_STR, "summary（情感状态事件摘要）："),

            ("emotional.events.*.experiencer", False, IS_STR, "experiencer（情绪经历主体）："),
            ("emotional.events.*.intensity", False, IS_FLOAT, "intensity（情绪）："),
            ("emotional.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("emotional.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化情感状态语义标识）："),
            ("emotional.events.*.expression_mode", False, IS_STR, "expression_mode（情绪表达模式）："),
            ("emotional.events.*.emotion_labels", False, IS_LIST, "emotion_labels（情绪标签）："),
            ("emotional.events.*.valence", False, IS_FLOAT, "valence（情绪效价）："),
            ("emotional.events.*.arousal", False, IS_FLOAT, "arousal（情绪唤醒度）：")
        ],

        # ────────────────────────────────────────
        # 13. 社会关系
        # ────────────────────────────────────────
        LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION: [
            ("social_relation", False, IS_DICT, "social_relation（社会关系信息根对象）："),
            ("social_relation.events", False, IS_LIST, "events（社会关系事件列表）："),
            ("social_relation.evidence", False, IS_LIST, "evidence（全局社会关系证据片段）："),
            ("social_relation.summary", False, IS_STR, "summary（社会关系事件摘要）："),

            ("social_relation.events.*.experiencer", False, IS_STR, "experiencer（社会关系经历主体）："),
            ("social_relation.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化社会关系语义标识）："),
            ("social_relation.events.*.relation_type", False, IS_STR, "relation_type（关系类型描述）："),
            ("social_relation.events.*.participants", False, IS_LIST, "participants（涉及的所有参与者）："),
            ("social_relation.events.*.source", False, IS_STR, "source（关系发起方）："),
            ("social_relation.events.*.target", False, IS_STR, "target（关系接收方）："),
            ("social_relation.events.*.confidence", False, IS_FLOAT, "confidence（关系置信度）："),
            ("social_relation.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）：")
        ],

        # ────────────────────────────────────────
        # 14. 合理推演
        # ────────────────────────────────────────
        LLM_INFERENCE: [
            ("inference", False, IS_DICT, "inference（合理推演信息根对象）："),
            ("inference.events", False, IS_LIST, "events（合理推理事件列表）："),
            ("inference.evidence", False, IS_LIST, "evidence（全局合理推演证据片段）："),
            ("inference.summary", False, IS_STR, "summary（合理推演事件摘要）："),

            ("inference.events.*.experiencer", False, IS_STR, "experiencer（推理主体）："),
            ("inference.events.*.inference_type", False, IS_STR, "inference_type（推理类型）："),
            ("inference.events.*.anchor_points", False, IS_LIST, "anchor_points（引用的感知事件 semantic_notation 列表）："),
            ("inference.events.*.inferred_proposition", False, IS_STR, "inferred_proposition（非确定性自然语言命题）："),
            ("inference.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("inference.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化合理推演语义标识）："),
            ("inference.events.*.polarity", False, IS_STR, "polarity（命题极性）："),
            ("inference.events.*.context_modality", False, IS_STR, "context_modality（语境模态）："),
            ("inference.events.*.scope", False, IS_STR, "scope（推理适用范围）：")
        ],

        # ────────────────────────────────────────
        # 15. 显性动机
        # ────────────────────────────────────────
        LLM_EXPLICIT_MOTIVATION_EXTRACTION: [
            ("explicit_motivation", False, IS_DICT, "explicit_motivation（显性动机信息根对象）："),
            ("explicit_motivation.events", False, IS_LIST, "events（显性动机事件列表）："),
            ("explicit_motivation.evidence", False, IS_LIST, "evidence（全局显性动机证据片段）："),
            ("explicit_motivation.summary", False, IS_STR, "summary（显性动机事件摘要）："),

            ("explicit_motivation.events.*.experiencer", False, IS_STR, "experiencer（陈述主体）："),
            ("explicit_motivation.events.*.evidence", False, IS_LIST, "evidence（事件级原文证据）："),
            ("explicit_motivation.events.*.semantic_notation", False, IS_STR, "semantic_notation（结构化显性动机语义标识）："),
            ("explicit_motivation.events.*.core_driver", False, IS_LIST, "core_driver（根本驱动力）："),
            ("explicit_motivation.events.*.care_expression", False, IS_LIST, "care_expression（关怀表达）："),
            ("explicit_motivation.events.*.separation_anxiety", False, IS_LIST, "separation_anxiety（分离焦虑）："),
            ("explicit_motivation.events.*.protective_intent", False, IS_LIST, "protective_intent（保护意图）："),

            ("explicit_motivation.events.*.power_asymmetry", False, IS_DICT, "power_asymmetry（权力不对称结构）："),
            ("explicit_motivation.events.*.power_asymmetry.control_axis", False, IS_LIST, "control_axis（控制方式）："),
            ("explicit_motivation.events.*.power_asymmetry.threat_vector", False, IS_LIST, "threat_vector（威胁手段）："),
            ("explicit_motivation.events.*.power_asymmetry.evidence", False, IS_LIST, "evidence（权力结构证据）："),

            ("explicit_motivation.events.*.resource_control", False, IS_LIST, "resource_control（资源控制）："),
            ("explicit_motivation.events.*.survival_imperative", False, IS_LIST, "survival_imperative（生存性服从）："),
            ("explicit_motivation.events.*.social_enforcement_mechanism", False, IS_LIST,
             "social_enforcement_mechanism（社会规范压力）："),

            ("explicit_motivation.events.*.narrative_distortion", False, IS_DICT, "narrative_distortion（话术策略）："),
            ("explicit_motivation.events.*.narrative_distortion.self_justification", False, IS_STR,
             "self_justification（自我合理化）："),
            ("explicit_motivation.events.*.narrative_distortion.blame_shift", False, IS_STR, "blame_shift（责任转嫁）："),
            ("explicit_motivation.events.*.narrative_distortion.moral_licensing", False, IS_STR,
             "moral_licensing（道德豁免）："),
            ("explicit_motivation.events.*.narrative_distortion.evidence", False, IS_LIST, "evidence（话术证据）："),
            ("explicit_motivation.events.*.internalized_oppression", False, IS_LIST, "internalized_oppression（内化压迫）："),
            ("explicit_motivation.events.*.motivation_category", False, IS_STR, "motivation_category（动机类型）：")
        ],

        # ────────────────────────────────────────
        # 16. 合理建议
        # ────────────────────────────────────────
        LLM_RATIONAL_ADVICE: [
            ("rational_advice", False, IS_DICT, "rational_advice（合理建议信息根对象）："),
            ("rational_advice.summary", False, IS_STR, "summary（合理建议事件摘要）："),
            ("rational_advice.semantic_notation", False, IS_STR, "semantic_notation（结构化合理建议语义标识）："),  # ← 用于跨模块关联
            ("rational_advice.evidence", False, IS_LIST, "evidence（全局合理建议证据片段）："),

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

# === 基线策略：严格模式===
STRICT_IRON_LAW_POLICY = {
    "context_isolation": True,
    "field_existence": "omit_if_absent",
    "literalism": True,
    "structure_consistency": True,
    "output_clean_json": True,
    "max_capture_min_fabrication": True,
    "enable_perception_rules": False
}

# === 步骤级策略覆盖表（按 step 名称定制）===
STEP_POLICY_OVERRIDES: Dict[str, Dict] = {
    # —————— 感知层（全部启用统一规则）——————
    "LLM_PERCEPTION_TEMPORAL_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_SPATIAL_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_VISUAL_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_AUDITORY_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_OLFACTORY_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_TACTILE_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_GUSTATORY_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_COGNITIVE_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_BODILY_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_EMOTIONAL_EXTRACTION": {"enable_perception_rules": True},
    "LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION": {"enable_perception_rules": True},

    # —————— 非感知层（特殊策略）——————
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
}


def get_effective_policy(step_name: str) -> Dict:
    """合并基线策略与步骤特例"""
    base = STRICT_IRON_LAW_POLICY.copy()
    override = STEP_POLICY_OVERRIDES.get(step_name, {})
    base.update(override)
    return base


def render_iron_law_from_policy(policy: Dict) -> str:
    """将策略字典渲染为自然语言铁律文本"""
    lines = ["### 【绝对铁律】"]

    if policy.get("context_isolation"):
        lines.append("1. 【上下文隔离原则】")
        lines.append("   - 完全无视历史对话与外部知识，禁止参考、延续或模仿任何过往输出内容，包括指令中的示例。")
    else:
        lines.append("1. 【上下文感知原则】")
        lines.append("   - 可安全访问已验证的上游输出作为当前任务的合法依据。")

    if policy.get("field_existence") == "omit_if_absent":
        lines.append("2. 【字段存在性总则】")
        lines.append("   - 字段仅在原文中存在直接、字面且语法关联的显式依据时方可输出。")
        lines.append("   - 无原文锚定的字段必须完全省略，禁止使用‘未提及’、‘未知’等占位符，亦不得以空字符串、null、空列表等形式表示。")

    if policy.get("literalism"):
        lines.append("3. 【字面主义至上】")
        lines.append("   - 所有属性值必须为原文中连续、字面、未改写的子字符串；")
        lines.append("     严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性。")
        lines.append("   - 无直接文字 = 无字段。")
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
        lines.append("   - 输出必须严格遵循指定 schema：字段名、类型、嵌套层级与序列格式均不得偏离。")
        lines.append("   - 所有列表字段若有值，须以非空列表形式（如 ['value']）返回；禁止使用裸字符串、数值、字典、null 或混合类型。")
        lines.append("   - 无有效值的字段（包括列表）必须完全省略，不得输出空列表或任何占位形式。")

    if policy.get("output_clean_json"):
        lines.append("5. 【输出洁净原则】")
        lines.append("   - 仅返回目标数据结构本身，无前缀、后缀、解释、注释、Markdown 或额外文本。")

    if policy.get("max_capture_min_fabrication"):
        lines.append("6. 【最大捕获最小编造原则】")
        if policy.get("literalism"):
            lines.append("   - 在严格字面约束下，穷尽所有可被直接锚定的语义单元；")
        else:
            lines.append("   - 在当前任务允许的表达范围内（如心理描写、自问句等），")
            lines.append("     穷尽所有可结构化的显式语义；")
        lines.append("   - 严禁以任何方式引入原文未出现的信息：包括但不限于常识推断、背景设定、角色动机、隐含关系、默认值或‘合理想象’。")
        lines.append("   - 所有输出必须可逐字回溯至 【唯一信源】；无法验证 = 不得存在。")
        lines.append("   - 目标：不错过，不编造。")

    if policy.get("enable_perception_rules", False):
        lines.append("7. 【感知层提取铁律】")
        lines.append("   - 【evidence 锚定】evidence 必须为原文中的连续子字符串，标点、大小写、数字格式完全一致；禁止 paraphrasing、概括、翻译、增删或改写。")
        lines.append("   - 【experiencer 提取优先级】按序确定：")
        lines.append("     • 优先提取与事件主语显式共指的具体 noun phrase（如同位语、前句主语、重复完整指称）；")
        lines.append("     • 若无具体 noun phrase，则提取代词，并标记为 '<代词>[uncertain]'；")
        lines.append("     • 仅当事件完全无主体（无人称句、纯客观陈述）且语法上无法推断感知主体时，才可省略 experiencer 字段。")
        lines.append("   - 【具体指称处理】experiencer 为具体 noun phrase 或专有名称时，直接逐字复制，不加任何标记。")
        lines.append("   - 【代词不确定性标记】以下情形必须标记 [uncertain]：")
        lines.append("     • 主语为代词（如'他''我'）且无明确共指对象；")
        lines.append("     • 主语为泛指（如'有人''一个人'）；")
        lines.append("     • 无法从上下文唯一确定指称的代词。")
        lines.append("   - 【标记完整性】禁止输出裸代词；所有代词 experiencer 必须包含 [uncertain] 标记。")

    return "\n".join(lines)
