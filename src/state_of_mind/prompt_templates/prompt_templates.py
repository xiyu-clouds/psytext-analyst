"""存放所有阶段模板定义"""

from src.state_of_mind.stages.perception.constants import CATEGORY_RAW, LLM_PARTICIPANTS_EXTRACTION, \
    LLM_PERCEPTION_TEMPORAL_EXTRACTION, LLM_PERCEPTION_SPATIAL_EXTRACTION, \
    LLM_PERCEPTION_VISUAL_EXTRACTION, LLM_PERCEPTION_GUSTATORY_EXTRACTION, LLM_PERCEPTION_TACTILE_EXTRACTION, \
    LLM_PERCEPTION_OLFACTORY_EXTRACTION, LLM_PERCEPTION_AUDITORY_EXTRACTION, LLM_PERCEPTION_EMOTIONAL_EXTRACTION, \
    LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION, SERIAL, PARALLEL, LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION, \
    LLM_PERCEPTION_COGNITIVE_EXTRACTION, LLM_PERCEPTION_BODILY_EXTRACTION, LLM_INFERENCE, LLM_RATIONAL_ADVICE, \
    PREPROCESSING, CATEGORY_SUGGESTION, LLM_EXPLICIT_MOTIVATION_EXTRACTION, COREFERENCE_RESOLUTION_BATCH, \
    LLM_DIMENSION_GATE, LLM_INFERENCE_ELIGIBILITY

LLM_PROMPTS_SCHEMA = {
    CATEGORY_RAW: {
        "version": "1.0.0",
        # === 控制流：处理阶段划分 ===
        "pipeline": [
            # 提取参与者
            {
                "step": LLM_PARTICIPANTS_EXTRACTION,
                "type": PREPROCESSING,
                "index": 0,
                "label": "预处理：大模型参与者信息提取",
                "role": "你是一个高保真结构化提取器。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                    "仅当【唯一信源】中对某 participant 有直接、字面、语法关联的描述时，才提取对应属性。无锚定 = 无字段。所有输出必须 100% 源自【唯一信源】区块。"
                ),
                "driven_by": "participants",
                "constraint_profile": "high_fidelity_participant_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【entity 核心定义与提取规则】——————
                        "- 【entity 提取优先级与合法性】按以下顺序确定 entity：",
                        "   • 优先提取完整、独立的名词短语，并逐字复制；",
                        "   • 若无上述具体指称，则提取出现的代词（如'我'、'他'）作为 entity；",
                        "   • 仅在完全无主体文本（无人称句）时，participants 数组才可为空。",
                        "- 【禁止模糊指称】禁止使用'那个身影'、'一个人'等模糊指称作为 entity，除非原文确无其他任何指称；",

                        # ——————【属性提取铁律】——————
                        "- 【禁止逆向推理】不得因某 entity 出现在某事件中，就自动赋予其事件角色（如'凶手''受害者'）；角色词必须原文显式出现，严禁任何形式的角色定性推断。",
                        "- 【appearance 视觉边界】appearance 字段仅接受对 entity 可被肉眼直接观测的外形、衣着、姿态、面部表情等字面描述；禁止填入背景故事、社会身份、职业标签、心理状态、修辞性比喻或任何非视觉信息。",
                        "- 【状态 vs 情绪边界】current_physical_state 仅接受可客观观测的身体状态或即时动作；所有情绪、心理感受、主观推测或由情绪引发的生理反应一律视为无效，必须省略。",
                        "- 【行为时效性边界】behavioral_tendencies 仅用于描述 entity 的长期、重复性、规律性行为模式或性格化习惯。",
                        "- 【持有动词边界】carried_objects 仅当原文中 entity 作为主语，且使用明确表示手持或随身携带的动词（如‘拿’、‘握’、‘揣’、‘拎’、‘抱’）时才可提取；物品仅出现在环境描述中（如‘信在桌上’）不构成携带。",

                        # ——————【实体管理规则】——————
                        "- 【实体隔离】每个 participant 的属性必须严格归属其自身 entity；禁止将其他 entity 的描述、动作或状态错误归因（例如：A 说‘我害怕’，不能将‘害怕’填入 B 的字段），即使上下文语义相关。",
                        "- 【实体合并条件】仅当多个 noun phrase 或代词在同一主语链或明确指代关系中（如‘一名警察……他……这名警官’），且共享关键特征（衣着、位置、行为）时，方可视为同一 entity 并合并；合并后 entity 名称必须使用首次出现的完整 noun phrase，不得自行概括或改写。",

                        # ——————【字段语义边界与存在性】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • entity：必须为原文中完整、独立的名词短语或代词",
                        "  • occupation：仅当原文明确使用职业身份词并直接指称该 entity 时才可提取",
                        "  • social_role：仅当原文使用非职业的社会身份标签并直接关联 entity 时才可提取",
                        "  • family_status：仅当原文使用家庭状态标签并直接陈述 entity 状态时才可提取",
                        "  • gender：仅当原文出现'男/女/男性/女性'等词并直接修饰或指称 entity 时才可提取；代词'他/她'不构成依据",
                        "  • age_range：仅当原文出现'青年/少年/中年/老年'等词并直接用于 entity 时才可提取",
                        "  • ethnicity_or_origin：仅当原文出现'来自[地点]''[地域]籍''生于[地]''[民族]族'等结构并直接关联 entity 时才可提取",
                        "  • physical_traits：仅当原文描述 entity 的固有、长期生理特征时才可提取",
                        "  • appearance：仅限视觉可观测的外貌描述；背景信息禁止填入",
                        "  • inherent_odor：仅当原文直接描述 entity 具有某种气味时才可提取",
                        "  • voice_quality：仅当原文直接描述 entity 的嗓音特征时才可提取",
                        "  • personality_traits：仅当原文使用评价性人格形容词并直接修饰或陈述 entity 时才可提取",
                        "  • behavioral_tendencies：仅限描述长期、重复性行为模式；单次行为必须省略",
                        "  • education_level：仅当原文出现明确教育程度表述并关联 entity 时才可提取",
                        "  • cultural_identity：仅当原文明确使用文化身份标签并指称 entity 时才可提取",
                        "  • primary_language：仅当原文陈述 entity 使用的主要语言时才可提取",
                        "  • institutional_affiliation：仅当原文提及 entity 所属组织时才可提取",
                        "  • current_physical_state：仅限字面生理/动作状态；心理状态禁止填入",
                        "  • visible_injury_or_wound：仅当原文明确描述 entity 身体上的可见伤痕、疤痕或医疗痕迹时才可提取",
                        "  • carried_objects：仅当原文包含物理持有动词且主语为 entity、宾语为具体物品时才可提取",
                        "  • worn_technology：仅当原文明确提及 entity 佩戴的电子设备时才可提取",
                        "  • speech_pattern：仅当原文明确提及口音、语速、用词习惯等时才可提取",
                        "  • interaction_role：仅当原文使用'凶手''目击者''受害者'等事件角色词并明确指派给 entity 时才可提取",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【participants 存在性规则】若存在至少一个合法 entity（包括代词），则 participants 数组必须非空；否则返回空数组 []"
                    ]
                },
                "fields": {
                    "participants": {
                        "type": "array",
                        "items": {
                            "entity": { "type": "string" },
                            "social_role": { "type": "string" },
                            "age_range": { "type": "string" },
                            "gender": { "type": "string" },
                            "ethnicity_or_origin": { "type": "string" },
                            "physical_traits": { "type": "array" },
                            "appearance": { "type": "array" },
                            "inherent_odor": { "type": "array" },
                            "voice_quality": { "type": "string" },
                            "personality_traits": { "type": "array" },
                            "behavioral_tendencies": { "type": "array" },
                            "education_level": { "type": "string" },
                            "occupation": { "type": "string" },
                            "family_status": { "type": "string" },
                            "cultural_identity": { "type": "array" },
                            "primary_language": { "type": "string" },
                            "institutional_affiliation": { "type": "array" },
                            "current_physical_state": { "type": "string" },
                            "visible_injury_or_wound": { "type": "array" },
                            "carried_objects": { "type": "array" },
                            "worn_technology": { "type": "array" },
                            "speech_pattern": { "type": "string" },
                            "interaction_role": { "type": "string" }
                        }
                    }
                }
            },

            # 动态决定启用哪些感知步骤
            {
                "step": LLM_DIMENSION_GATE,
                "type": PREPROCESSING,
                "role": (
                    "你是全息感知基底的预筛门控器，职责是在不启动任何感知解析器的前提下，"
                    "快速判断用户输入中是否包含可激活对应感知通道的显性语义线索。"
                ),
                "sole_mission": (
                    "【唯一信源】所有判断必须严格基于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；"
                    "系统指令、示例、模板等均不属于有效信源。\n"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，"
                    "可安全引用其中的字段定义与约束规则。\n"
                    "对照预定义的感知维度列表，为每个维度输出布尔值：true 表示存在字面或常规语义映射的显性线索，false 表示无。"
                ),
                "driven_by": "pre_screening",
                "constraint_profile": "high_fidelity_perceptual_gate_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        "每个字段的判断依据是：user_input 中是否包含属于该感知维度语义范畴的、直接且显性的语言表达。",
                        "显性表达指：无需任何推理、常识补全、语境整合或跨句关联即可识别的字面语义单元。",
                        "各维度的语义范畴定义如下：",
                        "- temporal: 对时间点、时间段、时序关系或时间状态的直接指涉。",
                        "- spatial: 对空间位置、方位、距离、布局或空间关系的直接描述。",
                        "- visual: 对可通过视觉直接观察的外显现象（包括身体动作、面部表情、衣着、颜色、物体状态或场景构图）的直接陈述。",
                        "- auditory: 对声音事件、语音特征、听觉感知或听觉缺失（如沉默）的直接陈述。",
                        "- olfactory: 对气味存在、性质或嗅觉行为的直接提及。",
                        "- tactile: 对触觉感受、物理接触、温度、质地或压力的直接描述。",
                        "- gustatory: 对味道、味觉体验或品尝行为的直接陈述。",
                        "- interoceptive: 对内部生理状态（如心跳、呼吸、肠胃、肌肉紧张等内感受信号）的直接觉察表述。",
                        "- cognitive: 对思维过程、记忆、判断、信念、困惑或意识活动的直接表达。",
                        "- bodily: 对非情绪性、非内感性的身体动作、姿态、状态或运动的直接描述，且不归属于 visual、auditory 或 interoceptive 范畴。",
                        "- emotional: 对情绪状态、情感反应或公认情绪行为的直接陈述，不包括对他人情绪的推测或隐喻性表达。",
                        "- social_relation: 对两个及以上参与者之间的身份、角色、关系类型或互动行为的直接提及。",
                        "关键原则：",
                        "• 若某维度的语义内容未以显性形式出现在 user_input 中，则必须标记为 false。",
                        "• 禁止将隐含、假设、反问、比喻、文化典故、模糊暗示或需外部知识理解的内容视为有效线索。",
                        "• 每个字段独立判断，互不影响；一个维度为 true 不导致其他维度自动为 true。",
                        "• 所有 true 判断必须可逐字回溯至 user_input 中的实际语义单元；无法定位 = false。"
                    ],
                    "output_structure_constraints": [
                        "【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "pre_screening 的值必须是一个对象，其字段名必须与 fields 块定义完全一致，值仅为 true 或 false。",
                        "禁止省略任何维度字段，禁止添加额外字段、注释、解释或格式装饰。",
                        "最终输出必须可通过 json.loads() 直接解析，无前缀后缀。"
                    ]
                },
                "fields": {
                    "pre_screening": {
                        "type": "object",
                        "properties": {
                            "temporal": {"type": "bool"},
                            "spatial": {"type": "bool"},
                            "visual": {"type": "bool"},
                            "auditory": {"type": "bool"},
                            "olfactory": {"type": "bool"},
                            "tactile": {"type": "bool"},
                            "gustatory": {"type": "bool"},
                            "interoceptive": {"type": "bool"},
                            "cognitive": {"type": "bool"},
                            "bodily": {"type": "bool"},
                            "emotional": {"type": "bool"},
                            "social_relation": {"type": "bool"}
                        }
                    }
                }
            },

            # 提前判断文本是否值得启用高阶推理
            {
                "step": LLM_INFERENCE_ELIGIBILITY,
                "type": PREPROCESSING,
                "role": (
                    "你是高阶认知流程的启动守门人，职责是判断当前用户输入是否包含足以激活‘合理推演-显性动机-合理建议’三步高阶推理链的显性语义基础。"
                ),
                "sole_mission": (
                    "【唯一信源】所有判断必须严格基于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；"
                    "系统指令、示例、模板等均不属于有效信源。\n"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，"
                    "可安全引用其中的字段定义与约束规则。\n"
                    "若输入中存在任一高阶推理所需的显性语义基础，则输出 eligible=true；否则为 false。"
                ),
                "driven_by": "eligibility",
                "constraint_profile": "inference_eligibility_abstract_strict_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        "eligible = true 当且仅当 USER_INPUT 中显性包含以下任一类语义内容：",
                        "- 对心理状态（如意图、信念、情绪、困惑、需求、恐惧）的直接陈述；",
                        "- 对行为目的、动机或理由的直接归因；",
                        "- 对两个及以上参与者之间互动、关系、角色或冲突的直接描述；",
                        "- 对决策困境、求助信号、犹豫、后悔或未来行动意向的直接表达。",
                        "显性指：该语义内容以字面形式出现在 USER_INPUT 中，无需任何推理、常识补全、语境整合、跨句关联或外部知识即可识别。",
                        "若 user_input 中不存在上述任何一类显性语义内容，则 eligible = false。",
                        "所有判断必须可逐字回溯至 user_input 中的实际文本片段；无法定位具体语义单元 = false。"
                    ],
                    "output_structure_constraints": [
                        "【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "输出必须包含且仅包含一个顶层字段：eligibility，其值为一个对象，包含唯一字段 eligible，值为 true 或 false。"
                    ]
                },
                "fields": {
                    "eligibility": {
                        "type": "object",
                        "properties": {
                            "eligible": {"type": "bool"}
                        }
                    }
                }
            },

            # 并行步骤
            # 时间感知
            {
                "step": LLM_PERCEPTION_TEMPORAL_EXTRACTION,
                "type": PARALLEL,
                "index": 1,
                "label": "感知层：大模型时间感知提取",
                "role": "你是一个严格遵循结构契约的时间信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "temporal",
                "constraint_profile": "high_fidelity_temporal_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【时间事件核心原则】——————
                        "- 【事件共现要求】每个时间事件必须同时包含：",
                        "    • 一个显式时间表达；",
                        "    • 至少一个 event_marker（动词或关键词）；",
                        "  二者须在同一最小语法单元中共现。纯时间短语无谓词不得提取；但含显式系动词或存在动词的孤立时间陈述视为有效。",
                        "- 【时间表达合法性】仅提取字面显式出现的时间表达，且必须同时满足：",
                        "    • 含时间单位词或文化/制度性时间标识（如'年''季度''春节''工作日'）；",
                        "    • 含具体数值、历法名称或标准化标签（如'2025''农历三月''Q3'）。",
                        "  孤立模糊时间词（如'最近''某天''以前'）不得提取。",
                        "- 【否定时间处理】被显式否定的时间表达，其时间成分必须提取至 negated_time 字段；否定词本身不提取，也不参与其他字段构建。",
                        "- 【复合时间解耦】仅当原文显式使用多个**独立且语法分离的时间成分**（如'2024年和25年''上午及晚上''周一至周三'）时，才分别提取各成分；  ",
                        "  若时间表达为**单一连续短语**（如'2024年3月15日''上周五下午'），即使包含多级单位，也视为整体，不得拆分； ",
                        "  禁止任何形式的分解、重组、推导或结构改写。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • evidence：必须为包含时间表达及其共现 event_marker 的最小连续原文片段；每个事件至少一个 evidence；允许多个。",
                        "  • semantic_notation：每个非空事件必须包含此字段；",
                        "    - 格式为 temporal_{time_category}_{semantic_feature}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
                        "    - time_category ∈ [absolute, relative, duration, frequency, range, cultural, negated]；",
                        "    - semantic_feature ∈ [point, span, future, past, continuous, recurring, boundary]；",
                        "    - english_summary 必须是一句高度提炼的英文事件概括，准确表达该时间事件的核心语义；",
                        "    - 该概括所依赖的所有事实要素（动作、对象、否定、时间关系等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、具体时间值、中文、拼音或泛化占位词；",
                        "    - 若无法生成合规摘要，则使用 temporal_event。"
                        "  • exact_literals：仅当原文出现包含具体数值与标准时间单位组合的精确时间字面量时才可提取",
                        "  • relative_expressions：仅当原文出现包含相对标记并与显式参考点共现的相对时间表达时才可提取",
                        "  • negated_time：仅当时间表达被显式否定动词/副词修饰时，提取被否定的时间成分",
                        "  • time_ranges：仅当原文出现明确起止结构的时间区间（含连接词如'到''至'）时才可提取",
                        "  • durations：仅当原文提及持续时长（含持续性动词或量词结构）时才可提取",
                        "  • frequencies：仅当原文出现周期性或频率表达（含频率副词或周期单位）时才可提取",
                        "  • event_markers：仅提取与时间表达共现于同一句法单元的动词或名词关键词",
                        "  • tense_aspect：仅当原文出现显式时体标记且直接修饰事件动词时才可提取",
                        "  • seasonal_or_cultural_time：仅当原文出现制度性或文化共识的时间标识时才可提取",
                        "  • temporal_anchor：仅当原文明确指定相对时间的参考点且该参考点为显式名词短语时才可提取",
                        "  • uncertainty_modifiers：仅当原文出现表示不确定性的修饰语且直接修饰时间成分时才可提取",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【temporal 存在性规则】",
                        "    • 若 events 为空，则 temporal 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 temporal 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                  },
                "fields": {
                    "temporal": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "experiencer": {"type": "string"},
                                        "evidence": {"type": "array"},
                                        "semantic_notation": {"type": "string"},
                                        "exact_literals": {"type": "array"},
                                        "relative_expressions": {"type": "array"},
                                        "negated_time": {"type": "array"},
                                        "time_ranges": {"type": "array"},
                                        "durations": {"type": "array"},
                                        "frequencies": {"type": "array"},
                                        "event_markers": {"type": "array"},
                                        "tense_aspect": {"type": "string"},
                                        "seasonal_or_cultural_time": {"type": "array"},
                                        "temporal_anchor": {"type": "string"},
                                        "uncertainty_modifiers": {"type": "array"},
                                    }
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"}
                        }
                    }
                }
            },

            # 空间感知
            {
                "step": LLM_PERCEPTION_SPATIAL_EXTRACTION,
                "type": PARALLEL,
                "index": 2,
                "label": "感知层：大模型空间感知提取",
                "role": "你是一个严格遵循结构契约的空间信息提取引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "spatial",
                "constraint_profile": "high_fidelity_spatial_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【空间事件核心原则】——————
                        "- 【空间锚定强制】每个空间事件必须包含至少一个显式空间锚定成分，满足以下任一条件：",
                        " • 含空间关系词；",
                        " • 含方向/朝向描述；",
                        " • 含空间介词；",
                        " • 含隐含定位的动词。",
                        "纯地点名词无上述上下文不得提取。",
                        "- 【事件共现要求】空间描述必须与至少一个 spatial_event_markers（动词或关键词）在同一最小语法单元中共现。孤立地点短语无谓词不得提取；但含显式系动词或存在动词的陈述视为有效。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • evidence：必须为包含空间描述及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、概括或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个有效空间事件必须包含此字段；",
                        "    - 格式为 spatial_{spatial_category}_{spatial_relation}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
                        "    - spatial_category 必须为以下之一：absolute, relative, directional, topological, cultural, negated；",
                        "    - spatial_relation 必须为以下之一：point, region, proximity, containment, boundary, path, origin, destination；",
                        "    - english_summary 必须是一句高度提炼的英文空间事件概括，准确表达该空间事件描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（实体、关系、方位、否定、参照物等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、坐标值、中文、拼音、系统提示占位词或模糊泛化标签；",
                        "    - 若无法生成合规摘要，则使用 spatial_event。",
                        "  • places：仅当原文明确提及具体场所名称时才可提取；抽象或模糊地点禁止提取。",
                        "  • layout_descriptions：仅当原文描述空间结构或排列方式时才可提取；必须为原文中的连续子字符串。",
                        "  • negated_places：仅当地点被显式否定时，提取被否定的地点名词；否定词本身不提取。",
                        "  • spatial_event_markers：仅提取与空间描述共现于同一句法单元的动词或名词关键词；必须是原文词汇，不得概括或替换。",
                        "  • cultural_or_institutional_spaces：仅当原文提及具有制度性或文化意义的空间单位并直接用于定位时才可提取。",
                        "  • orientation_descriptions：仅当原文出现明确方向或朝向描述时才可提取；必须包含方向词与参照物。",
                        "  • proximity_relations：仅当原文显式描述两个实体间的空间关系时才可构建该对象；其子字段规则如下：",
                        "    - actor：必须为关系中的主动方或参照主体，且为原文中显式 noun phrase；",
                        "    - target：必须为关系中的目标方或被参照对象，且为原文中显式 noun phrase；",
                        "    - relation_type：必须原样保留原文中的空间关系短语，禁止归一化、翻译或近义替换；",
                        "    - distance_cm：仅当原文明确给出数值与单位时提取，并统一换算为整数厘米；无显式数值则不得出现该字段；",
                        "    - modifiers：仅当原文出现修饰空间关系或感知条件的显式成分且与空间描述句法共现时才可提取；每个元素必须为原文中的连续子字符串；禁止基于常识推断媒介、阻碍物或环境。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【spatial 存在性规则】",
                        "    • 若 events 为空，则 spatial 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 spatial 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "spatial": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "experiencer": {"type": "string"},
                                        "evidence": {"type": "array"},
                                        "semantic_notation": {"type": "string"},
                                        "places": {"type": "array"},
                                        "layout_descriptions": {"type": "array"},
                                        "negated_places": {"type": "array"},
                                        "spatial_event_markers": {"type": "array"},
                                        "cultural_or_institutional_spaces": {"type": "array"},
                                        "orientation_descriptions": {"type": "array"},
                                        "proximity_relations": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "actor": {"type": "string"},
                                                    "target": {"type": "string"},
                                                    "distance_cm": {"type": "integer"},
                                                    "modifiers": {"type": "array"},
                                                    "relation_type": {"type": "string"},
                                                },
                                            }
                                        }
                                    },
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 视觉感知
            {
                "step": LLM_PERCEPTION_VISUAL_EXTRACTION,
                "type": PARALLEL,
                "index": 3,
                "label": "感知层：大模型视觉感知提取",
                "role": "你是一个严格遵循结构契约的视觉信息提取引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "visual",
                "constraint_profile": "high_fidelity_visual_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【视觉事件核心原则】——————
                        "- 【视觉事件锚定强制】每个视觉事件必须包含至少一个显式视觉成分，满足以下任一条件：",
                        "    • 视觉谓词（如'看见''闪烁''遮挡'）；",
                        "    • 视觉属性（如颜色、表情、姿态、纹理、透明度）；",
                        "    • 视觉媒介或状态描述（如'在监控中''模糊''反光'）。",
                        "  孤立名词若无共现的修饰成分（如形容词、动词、介词结构等），不得视为有效事件。",
                        "- 【事件共现要求】视觉描述必须与至少一个相关对象或动作在同一最小语法单元中共现。纯名词短语无上下文不得提取；但含显式系动词或感知动词的陈述视为有效。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】所有属性字段仅在其明确定义条件下提取；禁止跨字段挪用、泛化解释、默认填充或常识推理。无原文显式、直接、字面依据的字段必须彻底省略。",
                        "  • experiencer：必须为感知动词（如'看见''注意到'）或感知性系表结构（如'看起来''显得'）的主语，且为原文连续子字符串；转述句中 experiencer 为转述者（如'她说…' → '她'）。",
                        "  • observed_entity：仅限被直接感知的有生命主体（人/动物）或带明确指称的具体实体；须与感知动词或视觉属性在同一最小语法单元中共现；自身指称（如'自己''他的脸'）若显式出现可提取；禁止动作、状态、情绪、心理过程或抽象概念。",
                        "  • visual_objects：仅限无生命物体，且为原文显式提及的可见物；不得从动作隐含推导。",
                        "  • visual_attributes：仅提取原文中直接出现的颜色、形状、姿态、状态等描述短语；必须为连续子字符串，禁止拆分或重组。",
                        "  • visual_actions：仅当原文出现可见动作或姿态动词/短语（如'挥手''蜷缩'）时提取；必须为原文词汇，不得概括。",
                        "  • gaze_target：仅当原文显式包含注视动词（如'盯着''看向'）+目标名词时提取；无动词则不得提取。",
                        "  • eye_contact：仅当出现明确眼神交互描述（如'四目相对''避开视线'）时提取；泛化表达（如'看他一眼'）无效。",
                        "  • facial_cues：仅提取显式面部表情或微表情描述（如'皱眉''嘴角上扬'）；心理状态词（如'愤怒''悲伤'）禁止提取，除非以视觉形式出现（如'露出愤怒的表情'）。",
                        "  • salience：仅当原文出现确定性修饰词（如'清楚地''模糊地''隐约''完全看不见'）时才可量化；映射规则：'清楚地'→1.00，'隐约'→0.50，'模糊地'→0.30，'完全看不见'→0.00；无此类词则省略该字段。",
                        "  • negated_observations：仅当视觉内容被显式否定时，提取被否定的核心对象名词；否定词本身不提取。",
                        "  • visual_medium：仅当原文提及视觉媒介且与观察行为共现时才可提取；必须为原文连续片段。",
                        "  • occlusion_or_obstruction：仅当原文显式描述视线被阻挡时，提取遮挡物名词；必须为原文中实际出现的阻碍成分。",
                        "  • lighting_conditions：仅当原文明确描述光照环境时才可提取；必须为原文连续子字符串。",
                        "  • evidence：必须为包含整个视觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效视觉字段的事件必须包含此字段；",
                        "    - 格式为 visual_{visual_category}_{visual_feature}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
                        "    - visual_category 必须为以下之一：appearance, spatial, action, state, object, negated；",
                        "    - visual_feature 必须为以下之一：color, texture, position, arrangement, motion, static, onset, disappearance, visible, occluded；",
                        "    - english_summary 必须是一句高度提炼的英文视觉事件概括，准确表达该视觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（主体、对象、动作、属性、否定、媒介等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词（如 'angry'）、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 visual_event。"
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【visual 存在性规则】",
                        "    • 若 events 为空，则 visual 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 visual 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "visual": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "observed_entity": {"type": "string"},
                                    "visual_objects": {"type": "array"},
                                    "visual_attributes": {"type": "array"},
                                    "visual_actions": {"type": "array"},
                                    "gaze_target": {"type": "string"},
                                    "eye_contact": {"type": "array"},
                                    "facial_cues": {"type": "array"},
                                    "salience": {"type": "float"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                                    "negated_observations": {"type": "array"},
                                    "visual_medium": {"type": "array"},
                                    "occlusion_or_obstruction": {"type": "array"},
                                    "lighting_conditions": {"type": "array"},
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 听觉感知
            {
                "step": LLM_PERCEPTION_AUDITORY_EXTRACTION,
                "type": PARALLEL,
                "index": 4,
                "label": "感知层：大模型听觉感知提取",
                "role": "你是一个严格遵循结构契约的听觉信息提取引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "auditory",
                "constraint_profile": "high_fidelity_auditory_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【听觉事件核心原则】——————
                        "- 【听觉事件锚定强制】每个听觉事件必须包含至少一个显式听觉成分，满足以下任一条件：",
                        "   • 听觉谓词（如'听见''喊叫''低语'）；",
                        "   • 声音内容或非语言发声（如'脚步声''哭声''咔哒声'）；",
                        "   • 声音属性（如强度'大声'、节奏'急促'、停顿'沉默片刻'）；",
                        "   • 环境声或媒介描述（如'在电话里'）。",
                        "孤立声音名词若无共现修饰语、谓词或上下文支撑，不得视为有效事件。",
                        "【事件共现要求】听觉描述必须与至少一个相关成分（如声源、内容、强度、媒介或感知动词）在同一最小语法单元中共现。纯声音名词短语无上下文不得提取；但含显式感知动词的陈述视为有效。",

                        # ——————【听觉事件排除清单】——————
                        "- 【严格排除非物理听觉内容】即使出现'声音''听'等字眼，以下情形一律不得提取为听觉事件：",
                        "    • 内在心理活动：如'脑子里有个声音说…''内心独白''自我对话'；",
                        "    • 隐喻或抽象表达：如'良心的声音''时代的噪音''沉默的声音'；",
                        "    • 纯情绪或躯体反应：如'心慌''手心出汗''吼完很后悔''静不下来'。",
                        "- 上述内容属于 cognitive 或 interoceptive 事件范畴，禁止进入 auditory 模块。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】所有属性字段仅在其明确定义条件下提取；禁止跨字段挪用、泛化解释、默认填充或常识推理。无原文显式、直接、字面依据的字段必须彻底省略。",
                        "  • experiencer：必须为听觉动词（如'听见''听到''听出'）或听觉性表达（如'耳边响起''传来…声'）的主语，且为原文连续子字符串；转述句中 experiencer 为转述者（如'她说听见敲门声' → '她'）；自我感知若显式出现（如'我听见自己说话'），则 experiencer 为'我'。",
                        "  • sound_source：仅当原文显式指出声音的发出者或物理来源（如‘孩子哭’‘手机响了’‘风声呼啸’）时才可提取；必须为与声音描述共现的 noun phrase；禁止将声音内容、媒介或抽象概念作为 sound_source（如‘电话里的声音’中，sound_source 应为‘电话’而非‘声音’；若原文只说‘有声音’，则不得提取 sound_source）。",
                        "  • auditory_content：仅用于可转录的语言内容，包括直接引语或明确转述；必须为原文中实际出现的言语片段或关键词；非语言发声禁止填入此字段。",
                        "  • is_primary_focus：仅当原文出现显式焦点表达（如'只听到''注意力全在'）时才设为 true；否则必须省略该字段。",
                        "  • prosody_cues：仅提取原文显式描述的声音特征；每个元素必须为连续子字符串；心理状态词禁止提取，除非以声音形式出现（如'带着哭腔'）。",
                        "  • pause_description：仅当原文显式描述言语中的停顿时才可提取；必须为原文连续片段；环境静默不属于 pause。",
                        "  • intensity：仅当原文出现声音强度修饰词时才可量化；映射规则：'震耳欲聋'→1.00，'大声'→0.80，'正常'→0.60，'低声'→0.30，'隐约'→0.20，'微弱'→0.10；无此类词则省略该字段。",
                        "  • negated_observations：仅当听觉内容被显式否定时，提取被否定的核心对象或声音描述；否定词本身不提取。",
                        "  • auditory_medium：仅当原文提及听觉媒介且与声音共现时才可提取；必须为原文连续子字符串。",
                        "  • background_sounds：仅当原文明确描述环境背景声时才可提取；必须为原文中实际出现的环境声描述；不得从场景推断。",
                        "  • nonverbal_sounds：仅用于显式提及的非语言发声；必须为原文词汇；不得将语言内容误归入此字段。",
                        "  • evidence：必须为包含整个听觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效听觉要素的事件必须包含此字段；",
                        "    - 格式为 auditory_{sound_type}_{auditory_feature}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - sound_type 必须为以下之一：speech, sound_event, music, silence, negated；",
                        "    - auditory_feature 必须为以下之一：onset, duration, volume_high, volume_low, directional, repeating, abrupt, continuous, source_visible；",
                        "    - english_summary 必须是一句高度提炼的英文听觉事件概括，准确表达该听觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（声源、内容、强度、媒介、否定、环境等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词（如 'scared'）、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 auditory_event。"
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【auditory 存在性规则】",
                        "    • 若 events 为空，则 auditory 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 auditory 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "auditory": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "sound_source": {"type": "string"},
                                    "auditory_content": {"type": "array"},
                                    "is_primary_focus": {"type": "boolean"},
                                    "prosody_cues": {"type": "array"},
                                    "pause_description": {"type": "string"},
                                    "intensity": {"type": "float",},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                                    "negated_observations": {"type": "array"},
                                    "auditory_medium": {"type": "array"},
                                    "background_sounds": {"type": "array"},
                                    "nonverbal_sounds": {"type": "array"},
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 嗅觉感知
            {
                "step": LLM_PERCEPTION_OLFACTORY_EXTRACTION,
                "type": PARALLEL,
                "index": 5,
                "label": "感知层：大模型嗅觉感知提取",
                "role": "你是一个严格遵循结构契约的嗅觉信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "olfactory",
                "constraint_profile": "high_fidelity_olfactory_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【嗅觉事件核心原则】——————
                        "【嗅觉事件锚定强制】每个嗅觉事件必须包含至少一个显式气味描述，满足以下任一条件：",
                        "   • 具体气味词（如'臭味''花香''烟味''腐烂味'）；",
                        "   • 气味强度或情感修饰（如'刺鼻''淡淡''令人作呕''清香'）。",
                        "无具体气味内容的孤立表述（如'他闻了一下'）不得视为有效事件。",
                        "【事件共现要求】气味描述必须与至少一个相关成分（如气味来源、强度、感知动词或情感反应）在同一最小语法单元中共现。仅含动词'闻'而无气味词的结构不得提取。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】所有属性字段仅在其明确定义条件下提取；禁止跨字段挪用、泛化解释、默认填充或常识推理。无原文显式、直接、字面依据的字段必须彻底省略。",
                        "  • experiencer：必须为嗅觉动词（如'闻到''嗅到'）或气味感知表达（如'一股…味扑面而来''空气中弥漫着…'）的主语，且为原文连续子字符串；转述句中 experiencer 为转述者（如'他说闻到烟味' → '他'）；自我感知若显式出现（如'我闻到自己身上有汗味'），则 experiencer 为'我'。",
                        "  • odor_source：仅当原文显式指出气味的物理或生物来源（如‘饭菜香’‘垃圾发臭’‘她身上有香水味’）时才可提取；必须为与气味描述共现的 noun phrase；禁止将气味词、情感评价或抽象概念作为 odor_source（如‘一股臭味’中若未提来源，则不得提取 odor_source；‘香水味’不等于‘香水’，除非原文说‘香水的味道’或‘香水散发香味’）。",
                        "  • odor_descriptors：必须为原文直接出现的气味词或短语；每个元素必须为连续子字符串；禁止基于常识推断隐含气味。",
                        "  • intensity：仅当原文出现气味强度修饰词时才可量化；映射规则：'浓烈'/'刺鼻'→1.00，'扑鼻'→0.80，'明显'→0.60，'淡淡'→0.30，'微弱'→0.20；无此类词则省略该字段。",
                        "  • negated_observations：仅当嗅觉内容被显式否定时，提取被否定的核心对象或气味描述；否定词本身不提取。",
                        "  • odor_valence：仅当原文出现明确情感/评价词且与气味共现时才可提取；必须为原文词汇；不可从上下文情绪、表情或动作推导。",
                        "  • odor_source_category：仅当原文显式提及气味来源的大类时才可提取；必须为原文连续片段；禁止使用模糊词、专业术语或未出现的类别标签。",
                        "  • olfactory_actions：仅当原文出现显式嗅觉相关动作且与气味描述共现时才可提取；必须为原文连续子字符串；动词'闻'本身不构成动作细节。",
                        "  • evidence：必须为包含整个嗅觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效嗅觉要素的事件必须包含此字段；",
                        "    - 格式为 olfactory_{odor_category}_{perception_feature}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - odor_category 必须为以下之一：odor, scent, stench, chemical, negated；",
                        "    - perception_feature 必须为以下之一：presence, absence, intensity_high, intensity_low, diffusion, localized, onset；",
                        "    - english_summary 必须是一句高度提炼的英文嗅觉事件概括，准确表达该嗅觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（气味词、来源、强度、情感、否定、动作等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 olfactory_event。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【olfactory 存在性规则】",
                        "    • 若 events 为空，则 olfactory 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 olfactory 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "olfactory": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "description": "气味感知者"
                                    },
                                    "odor_source": {
                                        "type": "string",
                                        "description": "气味来源"
                                    },
                                    "odor_descriptors": {
                                        "type": "array",
                                        "description": "显式出现的气味描述词或短语"
                                    },
                                    "intensity": {
                                        "type": "float",
                                        "description": "仅当有强度修饰词时量化"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "description": "evidence 必须是用户输入中的连续子字符串（可通过 substring 检查验证）。允许保留原文所有字符（包括标点、数字、大小写）；禁止改写、概括、翻译或增删实质性词汇。可忽略前导/尾随空白符及换行符差异。"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "description": (
                                            "【硬性规则】\n"
                                            "- 若该事件包含任一有效嗅觉要素（如 odor_descriptors、negated_observations、olfactory_actions 等），则必须输出此字段\n"
                                            "- 格式：{odor_category}_{perception_feature}_{odor_core}（总长 ≤128，snake_case）\n"
                                            "- odor_category ∈ [odor, scent, stench, chemical, negated]\n"
                                            "- perception_feature ∈ [presence, absence, intensity_high, intensity_low, diffusion, localized, onset]\n"
                                            "- odor_core 必须使用以下预定义英文标签之一（不可自造）：\n"
                                            "    • gas_leak              （如‘煤气味’）\n"
                                            "    • blood                 （如‘血腥味’）\n"
                                            "    • rotten_egg            （如‘臭鸡蛋味’）\n"
                                            "    • sandalwood            （如‘檀香’）\n"
                                            "    • floral                （如‘花香’）\n"
                                            "    • mold                  （如‘霉味’）\n"
                                            "    • smoke                 （如‘烟味’）\n"
                                            "    • perfume               （如‘香水味’）\n"
                                            "    • food_cooking          （如‘饭菜香’）\n"
                                            "    • chemical_cleaner      （如‘消毒水味’）\n"
                                            "- 若气味描述不属于以上预定义组合，则 odor_core 使用 generic_odor（泛气味）、generic_scent（泛香气）、generic_stench（泛恶臭）或 olfactory_event（抽象嗅觉事件）\n"
                                            "- 所有组成部分必须严格对应输入文本中的显式词汇或上述预定义集合；禁止使用中文、拼音、人名、品牌、心理状态词（如 'disgust'）、动作动词（如 'smelled'）或未出现的复合词\n"
                                            "- 示例合法值：chemical_intensity_high_gas_leak, stench_presence_rotten_egg, negated_absence_perfume, scent_diffusion_floral"
                                        )
                                    },
                                    "negated_observations": {
                                        "type": "array",
                                        "description": "显式否定的嗅觉对象或描述"
                                    },
                                    "odor_valence": {
                                        "type": "array",
                                        "description": "显式出现的情感/评价词（如‘恶心’‘清香’）"
                                    },
                                    "odor_source_category": {
                                      "type": "array",
                                      "description": "显式提及的气味来源大类"
                                    },
                                    "olfactory_actions": {
                                        "type": "array",
                                        "description": "显式描述的嗅觉相关动作（如‘抽鼻子’‘捂住鼻子’）"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "description": "支撑整体嗅觉判断的原文片段（所有 events.evidence 的并集，去重）"
                            },
                            "summary": {
                                "type": "string",
                                "description": "summary 必须仅由 events 中显式出现的嗅觉要素重组而成，不得引入任何未在 evidence 中出现的词汇或逻辑连接（如‘因此’‘随后’）或隐含拓扑推论。仅允许使用中性连接词（如‘和’‘以及’‘还有’）以保证语句通顺"
                            }
                        }
                    }
                }
            },

            # 触觉感知
            {
                "step": LLM_PERCEPTION_TACTILE_EXTRACTION,
                "type": PARALLEL,
                "index": 6,
                "label": "感知层：大模型触觉感知提取",
                "role": "你是一个严格遵循结构契约的触觉信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "tactile",
                "constraint_profile": "high_fidelity_tactile_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【触觉事件核心原则】——————
                        "- 【触觉事件锚定强制】每个触觉事件必须同时包含：",
                        "   • 一个显式触觉对象（如'手''桌面''皮肤'等 body_part 或 contact_target）；",
                        "   • 一个具体触觉属性（如'冰冷''粗糙''剧痛''震动''刺痒'）。",
                        "无具体属性或对象的孤立表述（如'碰了一下''感觉不对'）不得视为有效事件。",
                        "- 【事件共现要求】触觉对象与属性必须在同一最小语法单元中共现。仅含动作动词（如'摸''碰''感到'）而无具体属性描述的结构不得提取。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：必须为原文中触觉感知动词（如‘感到’‘觉得’‘察觉到’）或触觉属性描述（如‘冰冷刺骨’‘粗糙扎手’）的主语；若为转述句（如‘她说手很疼’），则 experiencer 为转述者（‘她’）；若描述自身身体部位感受（如‘我手指发麻’），且原文显式出现，则 experiencer 为‘我’。",
                        "  • contact_target：仅当原文显式提及被接触或被感知的物体、表面或实体（如‘墙壁冰冷’‘沙子粗糙’‘刀刃割手’）时才可提取；必须为与触觉属性共现的 noun phrase；禁止将动作对象、施事者或抽象概念作为 contact_target（如‘他摸了一下’中若未提‘什么’，则不得提取 contact_target）。",
                        "  • tactile_descriptors：必须保留原文完整的触觉动词/短语；每个元素必须为连续子字符串；不可拆分动词与副词。",
                        "  • contact_initiator：仅当原文显式出现主动接触方且与 contact_target 构成主-宾关系时才可提取；必须为完整 noun phrase；不可从单方动作反推另一方。",
                        "  • body_part：仅当原文显式提及身体部位且与触觉描述共现时才可提取；必须为原文连续子字符串；不可默认填充。",
                        "  • tactile_intent_or_valence：仅当原文出现明确意图/情感修饰副词且修饰触觉动作时才可提取；必须为原文词汇；不可由动作类型或结果推导意图。",
                        "  • negated_observations：仅当触觉内容被显式否定时，提取被否定的核心对象或描述；否定词本身不提取。",
                        "  • texture：仅当原文出现质地描述词且与接触对象共现时才可提取；必须为原文连续片段；禁止从材料推断。",
                        "  • temperature：仅当原文出现温度描述词且与接触对象共现时才可提取；必须为原文词汇；不可从环境推断。",
                        "  • pressure：仅当原文出现压力相关描述且与触觉共现时才可提取；必须为原文连续子字符串。",
                        "  • pain：仅当原文出现疼痛类描述且与身体部位或接触共现时才可提取；必须为原文词汇；不可从表情或动作推导。",
                        "  • motion：仅当原文出现触觉动态模式描述且属于微观体感时才可提取；宏观物体运动不计入。",
                        "  • vibration：仅当原文直接出现震动类词汇时才可提取；必须为原文连续片段；不可从设备类型推断。",
                        "  • moisture：仅当原文出现湿度/干湿状态描述且与接触面共现时才可提取；不可从天气、场景隐含推断。",
                        "  • contact：仅当原文显式描述接触的物理状态（如“轻轻贴着”“隔着布料”）时才可提取；强调接触方式而非动作本身；必须为原文连续子字符串。",
                        "  • intensity：仅当原文出现强度修饰词时才可量化；映射规则：'剧烈'/'猛烈'/'撕心裂肺'→1.00，'明显'→0.70，'轻微'/'隐隐'→0.30，'柔和'→0.40；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个触觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效触觉要素的事件必须包含此字段；",
                        "    - 格式为 tactile_{tactile_modality}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - tactile_modality 必须为以下之一：temperature, texture, pressure, vibration, pain, moisture, contact, motion, negated；",
                        "    - english_summary 必须是一句高度提炼的英文触觉事件概括，准确表达该触觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（触觉词、对象、部位、强度、否定等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词（如 'comfort'）、动作动词（如 'touched'）、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 tactile_event。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【tactile 存在性规则】",
                        "    • 若 events 为空，则 tactile 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 tactile 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "tactile": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "contact_target": {"type": "string"},
                                    "tactile_descriptors": {"type": "array"},
                                    "contact_initiator": {"type": "string"},
                                    "body_part": {"type": "array"},
                                    "tactile_intent_or_valence": {"type": "array"},
                                    "negated_observations": {"type": "array"},
                                    "texture": {"type": "array"},
                                    "temperature": {"type": "array"},
                                    "pressure": {"type": "array"},
                                    "pain": {"type": "array"},
                                    "motion": {"type": "array"},
                                    "vibration": {"type": "array"},
                                    "moisture": {"type": "array"},
                                    "contact": {"type": "array"},
                                    "intensity": {"type": "float"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"}
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 味觉感知
            {
                "step": LLM_PERCEPTION_GUSTATORY_EXTRACTION,
                "type": PARALLEL,
                "index": 7,
                "label": "感知层：大模型味觉感知提取",
                "role": "你是一个严格遵循结构契约的味觉信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "gustatory",
                "constraint_profile": "high_fidelity_gustatory_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【味觉事件核心原则】——————
                        "- 【味觉事件锚定强制】每个味觉事件必须同时包含：",
                        "   • 一个显式味道来源（如'咖啡''药片''水果'等 taste_source）；",
                        "   • 一个具体味觉属性（如'甜''苦''酸''辣''涩''咸'）。",
                        " 未明确说明味道类型的孤立表述（如'尝了一口''有股怪味'）不得视为有效事件。",
                        "【事件共现要求】味道来源与味觉属性必须在同一最小语法单元中共现。仅含动作动词（如'尝''吃''觉得'）而无具体味觉词的结构不得提取。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：必须为原文中味觉动词（如‘尝到’‘觉得’‘感到’）或味觉属性描述（如‘甜得发腻’‘苦不堪言’）的主语；若为转述句（如‘他说咖啡很苦’），则 experiencer 为转述者（‘他’）；若描述自身摄入并感知味道（如‘我喝了一口汤，很咸’），且原文显式出现，则 experiencer 为‘我’。",
                        "  • taste_source：仅当原文显式提及被品尝的食物、饮品或物质（如‘咖啡很苦’‘药片发涩’‘西瓜真甜’）时才可提取；必须为与味觉属性共现的 noun phrase；禁止将抽象概念、动作对象或未指明的‘东西’作为 taste_source（如‘尝了点东西，很难吃’中若未说明‘什么’，则不得提取 taste_source）。",
                        "  • taste_descriptors：必须保留原文完整的味觉描述短语；每个元素必须为连续子字符串；不可拆分形容词与修饰语。",
                        "  • contact_initiator：仅当原文显式出现主动摄入方且与 taste_source 构成主-宾关系时才可提取；必须为完整 noun phrase；不可从单方动作反推另一方。",
                        "  • body_part：仅当原文指出具体味觉发生部位（如“舌尖”“喉咙”）时才可提取；必须为原文连续子字符串；禁止默认填充未出现词汇。",
                        "  • intent_or_valence：仅当原文出现明确情感或意图表达（如“嫌弃地吐掉”“陶醉地吮吸”）时才可提取；必须为原文连续副词/动词短语；不可由味道类型或食物反推情绪。",
                        "  • negated_observations：仅当味觉内容被显式否定时，提取被否定的核心对象或描述；否定词本身不提取。",
                        "  • sweet / salty / sour / bitter / umami / spicy / astringent / fatty / metallic / chemical / thermal：仅当原文出现对应味觉/口感/温度描述词且与 taste_source 共现时才可提取；必须为原文词汇；禁止由食物、成分或常识推断。",
                        "  • intensity：仅当原文出现强度修饰词时才可量化；映射规则：'极其'/'浓烈'/'强烈'→1.00，'明显'→0.70，'淡淡'/'微弱'/'隐约'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个味觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效味觉要素的事件必须包含此字段；",
                        "    - 格式为 gustatory_{gustatory_modality}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - gustatory_modality 必须为以下之一：sweet, salty, sour, bitter, umami, spicy, astringent, fatty, metallic, chemical, thermal, negated；",
                        "    - english_summary 必须是一句高度提炼的英文味觉事件概括，准确表达该味觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（味道词、来源、部位、强度、否定等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 gustatory_event。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【gustatory 存在性规则】",
                        "    • 若 events 为空，则 gustatory 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 gustatory 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "gustatory": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "taste_source": {"type": "string"},
                                    "taste_descriptors": {"type": "array"},
                                    "contact_initiator": {"type": "string"},
                                    "body_part": {"type": "string"},
                                    "intent_or_valence": {"type": "array"},
                                    "negated_observations": {"type": "array"},
                                    "sweet": {"type": "array"},
                                    "salty": {"type": "array"},
                                    "sour": {"type": "array"},
                                    "bitter": {"type": "array"},
                                    "umami": {"type": "array"},
                                    "spicy": {"type": "array"},
                                    "astringent": {"type": "array"},
                                    "fatty": {"type": "array"},
                                    "metallic": {"type": "array"},
                                    "chemical": {"type": "array"},
                                    "thermal": {"type": "array"},
                                    "intensity": {"type": "float"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 内感受
            {
                "step": LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION,
                "type": PARALLEL,
                "index": 8,
                "label": "感知层：大模型内感受感知提取",
                "role": "你是一个严格遵循结构契约的内感受信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "interoceptive",
                "constraint_profile": "high_fidelity_interoceptive_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【内感受事件核心原则】——————
                        "- 【内感受事件锚定强制】每个内感受事件必须同时包含：",
                        "   • 一个显式身体部位（如'胸口''手心''胃'）或明确触发条件；",
                        "   • 一个具体生理感受描述（如'心悸''腿软''口干''出汗''头晕'）。",
                        "仅含情绪词（如'害怕''紧张'）而无身体感受的表述不得视为有效事件。",
                        "【情绪 ≠ 生理】情绪词汇（如'焦虑''兴奋''恐惧'）不得视为生理感受；必须有字面出现的身体感觉短语才可提取。",
                        "【事件共现要求】身体部位/触发条件与生理感受必须在同一最小语法单元中共现。纯情绪陈述或孤立心理描述无身体词不得提取。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：必须为原文中生理感受描述（如‘心悸’‘腿软’‘口干’）或内感受动词（如‘感到’‘觉得’‘察觉到’）的主语；若为转述句（如‘他说胃很胀’），则 experiencer 为转述者（‘他’）；若描述自身生理状态（如‘我心跳加速’），且原文显式出现，则 experiencer 为‘我’。",
                        "  • contact_initiator：仅当原文显式提及引发生理感受的外部诱因、行为或条件（如‘跑完步后’‘闻到臭味时’‘看到血就头晕’）时才可提取；必须为与生理感受共现的连续子字符串；禁止将情绪、心理状态或未说明的因果关系作为 contact_initiator。",
                        "  • body_part：仅当原文明确指出感受发生的具体身体部位（如‘胸口’‘胃里’‘太阳穴’）且与生理感受共现时才可提取；必须为原文连续子字符串；禁止使用泛称（如‘身体’‘里面’‘全身’）或未出现的解剖术语。",
                        "  • intent_or_valence：仅当原文出现明确情感动作或副词修饰时才可提取；必须为原文连续片段；不可由感受类型反推情绪。",
                        "  • negated_observations：仅当内感受内容被显式否定时，提取被否定的核心对象或描述；否定词本身不提取。",
                        "  • cardiac / respiratory / gastrointestinal / thermal / muscular / visceral_pressure / dizziness / nausea / fatigue / thirst_hunger：仅当原文出现对应生理感受描述词且共现时才可提取；必须为原文词汇；禁止由情绪、动作或常识推断。",
                        "  • intensity：仅当原文出现强度修饰词时才可量化；映射规则：'剧烈'/'难以忍受'→1.00，'明显'/'持续'→0.70，'隐隐'/'轻微'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个内感受事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效内感受要素的事件必须包含此字段；",
                        "    - 格式为 interoceptive_{interoceptive_category}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - interoceptive_category 必须为以下之一：cardiac, respiratory, gastrointestinal, thermal, muscular, visceral_pressure, dizziness, nausea, fatigue, thirst_hunger, negated；",
                        "    - english_summary 必须是一句高度提炼的英文内感受事件概括，准确表达该生理感受的核心语义；",
                        "    - 该概括所依赖的所有关键要素（感受词、部位、诱因、强度、否定等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、医学术语、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 interoceptive_event。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【interoceptive 存在性规则】",
                        "    • 若 events 为空，则 interoceptive 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 interoceptive 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "interoceptive": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "contact_initiator": {"type": "string"},
                                    "body_part": {"type": "string"},
                                    "intent_or_valence": {"type": "array"},
                                    "negated_observations": {"type": "array"},
                                    "cardiac": {"type": "array"},
                                    "respiratory": {"type": "array"},
                                    "gastrointestinal": {"type": "array"},
                                    "thermal": {"type": "array"},
                                    "muscular": {"type": "array"},
                                    "visceral_pressure": {"type": "array"},
                                    "dizziness": {"type": "array"},
                                    "nausea": {"type": "array"},
                                    "fatigue": {"type": "array"},
                                    "thirst_hunger": {"type": "array"},
                                    "intensity": {"type": "float"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },
            },

            # 认知过程
            {
                "step": LLM_PERCEPTION_COGNITIVE_EXTRACTION,
                "type": PARALLEL,
                "index": 9,
                "label": "感知层：大模型认知过程感知提取",
                "role": "你是一个严格遵循结构契约的认知过程信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "cognitive",
                "constraint_profile": "high_fidelity_cognitive_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【认知事件核心原则】——————
                        "- 【认知事件成立条件】每个认知事件必须由显式认知动词（如'认为''记得''打算''怀疑''分析''意识到''自问'）或引语结构（如'心想''暗道''觉得'）引导，并包含完整的思维内容；孤立副词、行为描写或情绪状态（如'犹豫地走开''他很困惑'）不得视为有效认知事件。",
                        "- 【事件共现要求】认知动词与思维内容必须在同一最小语法单元中共现。仅含动作、表情或情绪而无显式思维内容的结构不得提取。",
                        "- 【多对象拆分原则】当单句中显式包含多个独立思维对象（由逗号、连词如'和''但''又'等分隔），且每个对象均有独立谓词或评价时，必须拆分为多个认知事件，确保每事件仅对应一个（experiencer, cognitive_agent, target_entity）三元组。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：必须为原文中认知动词或引语结构的主语；若为转述句，则 experiencer 为转述者（‘她’）；若描述自身思维，且原文显式出现，则 experiencer 为‘我’。",
                        "  • cognitive_agent：仅当认知内容明确转述他人观点或话语时才可提取；必须为被转述观点的原始发出者，且原文显式提及；若认知内容为自思、无转述结构，或未指明原说话人，则彻底省略该字段。",
                        "  • target_entity：必须为原文中与认知动词直接关联的**单一、具体、连续**的思维对象或主题；若认知内容涉及多个独立对象（由逗号、‘和’‘但’等连接），必须拆分为多个事件，每个事件仅含一个 target_entity；禁止合并、概括、使用代词或模糊指称；若无法确定唯一锚定对象，则彻底省略该字段。",
                        "  • cognitive_valence：仅当原文出现明确情感修饰语修饰认知动词（时才可提取；必须为原文连续片段；不可由上下文反推。",
                        "  • negated_cognitions：仅当认知内容被显式否定时，提取整个被否定的认知短语（保留原结构）；否定词本身不单独提取。",
                        "  • belief / intention / inference / memory_recall / doubt_or_uncertainty / evaluation / problem_solving / metacognition：仅当原文出现对应认知动词或结构引导的完整陈述时才可提取；必须为原文词汇；禁止由语气、结果或常识推断。",
                        "  • intensity：仅当原文出现确信度修饰词（如‘绝对’‘隐约觉得’）时才可量化；映射规则：'绝对'/'百分百'→1.00，'几乎'/'基本'→0.70，'隐约'/'有点'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个认知事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效认知要素的事件必须包含此字段；",
                        "    - 格式为 cognitive_{cognitive_category}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - cognitive_category 必须为以下之一：belief, intention, inference, memory_recall, doubt_or_uncertainty, evaluation, problem_solving, metacognition；",
                        "    - english_summary 必须是一句高度提炼的英文认知事件概括，准确表达该思维内容的核心语义；",
                        "    - 该概括所依赖的所有关键要素（动词、对象、修饰语、解决方案、怀疑点等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、代词、评价性形容词、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 cognitive_event。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【cognitive 存在性规则】",
                        "    • 若 events 为空，则 cognitive 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 cognitive 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "cognitive": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "cognitive_agent": {"type": "string"},
                                    "target_entity": {"type": "string"},
                                    "cognitive_valence": {"type": "array"},
                                    "negated_cognitions": {"type": "array"},
                                    "belief": {"type": "array"},
                                    "intention": {"type": "array"},
                                    "inference": {"type": "array"},
                                    "memory_recall": {"type": "array"},
                                    "doubt_or_uncertainty": {"type": "array"},
                                    "evaluation": {"type": "array"},
                                    "problem_solving": {"type": "array"},
                                    "metacognition": {"type": "array"},
                                    "intensity": {"type": "float"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 躯体化表现
            {
                "step": LLM_PERCEPTION_BODILY_EXTRACTION,
                "type": PARALLEL,
                "index": 10,
                "label": "感知层：大模型躯体化表现感知提取",
                "role": "你是一个严格遵循结构契约的躯体化表现信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "bodily",
                "constraint_profile": "high_fidelity_bodily_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【躯体行为核心原则】——————
                        "- 【可观测性强制】仅提取第三方可直接观测的物理行为；主观感受、内部生理状态或不可见心理活动一律排除。",
                        "- 【行为锚定】每个躯体行为必须为字面显式的物理动作或可见变化（如'皱眉''声音发抖''手心出汗'）；情绪副词（如'紧张地''愤怒地'）无伴随具体动作不得提取。",
                        "- 【动态与静态分离】若行为同时包含运动与姿态（如'慢慢蹲下并蜷缩'），movement_direction 提取运动部分（'蹲下'），posture 提取姿态部分（'蜷缩'）；若仅为静态姿态（如'弓着背坐着'），仅填 posture，不得虚构 movement_direction。",

                        # ——————【freeze_or_faint 字段特别警示】——————
                        "- 【freeze_or_faint 字段提取条件】仅当原文出现以下**字面显式、非隐喻**的描述时才可提取：",
                        "   **冻结行为**：身体完全或近乎静止，且用词直接表达运动丧失，如'僵住不动''一动不动''愣在原地''呆立''身体冻住''动弹不得''像被钉住''整个人定住'；",
                        "   **晕厥前兆**：明确描述即将失去意识或平衡的身体征兆，如'站立不稳''摇晃''眼前发黑''两眼发黑''天旋地转''站不住''腿软''差点晕倒''险些昏过去''要栽倒'。",
                        "- 【严格排除】以下情况即使语义接近也不得提取：",
                        "    • 主观感受（如'感觉要晕了''脑子空白'）；",
                        "    • 隐喻或夸张修辞（如'吓傻了''石化了'除非上下文明确指身体僵直）；",
                        "    • 纯自主神经症状（如'心悸''出汗''呼吸急促'）；",
                        "    • 未使用上述语义范畴内显式动词/短语的间接描述。",
                        "- 若无符合上述条件的字面表述，freeze_or_faint 字段必须省略。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：必须为原文中躯体行为动词（如‘皱眉’‘蹲下’‘声音发抖’）或可见状态描述（如‘手心出汗’‘弓着背’）的主语；若行为由他人转述（如‘他说她僵住了’），则 experiencer 为被描述的行为执行者（‘她’）；若为自我报告（如‘我腿软了’），且原文显式出现，则 experiencer 为‘我’。",
                        "  • observer：仅当躯体行为通过他人视角被转述，且观察者身份以完整 noun phrase 显式出现时才可提取（如‘护士注意到他颤抖’中的‘护士’）；若为客观叙述（如‘他颤抖’）、自我报告（如‘我出汗了’）或未指明观察者（如‘有人看见他晕倒’中的‘有人’为泛指），则 observer 必须省略。",
                        "  • movement_direction：仅当原文包含显式运动方向或位移动词时才可提取；必须为完整短语；静态描述不得填入此字段。",
                        "  • posture：仅当原文描述静态身体姿态时才可提取；若姿态伴随运动，则 posture 仅提取静态部分，movement_direction 提取动态部分。",
                        "  • facial_expression：仅当出现可观察的面部物理变化时才可提取；必须为连续子字符串；情绪标签不得提取。",
                        "  • vocal_behavior：仅当描述声音的物理特征时才可提取；内容性或纯情绪性描述不得提取。",
                        "  • autonomic_signs：仅当出现他人可见的自主神经系统外显反应时才可提取；内部感受不得提取；必须为可观测现象。",
                        "  • motor_behavior：仅当出现显式随意运动时才可提取；必须为具体动作；模糊描述若无具体动作支撑，不得提取。",
                        "  • freeze_or_faint：仅当出现显式冻结或晕厥倾向行为时才可提取；主观体验不得作为依据；必须为可观测的身体状态变化。",
                        "  • intensity：仅当原文出现强度修饰词时才可量化；映射规则：'剧烈'/'完全'→1.00，'明显'/'大幅'→0.70，'微微'/'轻轻'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个躯体行为及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效躯体化要素的事件必须包含此字段；",
                        "    - 格式为 bodily_{bodily_category}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - bodily_category 必须为以下之一：facial, vocal, postural, locomotor, manual, autonomic_visible, freeze；",
                        "    - english_summary 必须是一句高度提炼的英文躯体行为概括，准确表达该可观测行为的核心语义；",
                        "    - 该概括所依赖的所有关键要素（动作、部位、方向、强度、可见征象等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含情绪词、医学术语、内部感受、代词（如 'it', 'that'）、品牌、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 bodily_event。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【bodily 存在性规则】",
                        "    • 若 events 为空，则 bodily 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 bodily 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "bodily": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "observer": {"type": "string"},
                                    "movement_direction": {"type": "string"},
                                    "posture": {"type": "string"},
                                    "facial_expression": {"type": "array"},
                                    "vocal_behavior": {"type": "array"},
                                    "autonomic_signs": {"type": "array"},
                                    "motor_behavior": {"type": "array"},
                                    "freeze_or_faint": {"type": "array"},
                                    "intensity": {"type": "float"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"}
                        }
                    }
                }
                }
            },

            # 情感状态
            {
                "step": LLM_PERCEPTION_EMOTIONAL_EXTRACTION,
                "type": PARALLEL,
                "index": 11,
                "label": "感知层：大模型情感状态感知提取",
                "role": "你是一个严格遵循结构契约的情感状态信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "emotional",
                "constraint_profile": "high_fidelity_emotional_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【情绪提取核心原则】——————
                        "- 【情绪词字面强制】emotion_labels 必须为原文中逐字出现的情绪词或情绪修饰短语（如'悲痛欲绝地说''带着怒意''好烦啊'）；禁止替换、拆分、同义转换、抽象化、标准化或映射至任何情绪分类体系。",
                        "- 【行为 ≠ 情绪】哭泣、跺脚、沉默、叹气、脸红、转身、颤抖、握拳等行为、生理反应或躯体表现不得作为情绪依据；仅当字面出现情绪形容词、副词、名词或含情绪语义的固定短语时，才可提取 emotion_labels。",
                        "- 【混合语言保真】若原文混用中英文情绪表达（如'他很 jiaolv''她 feel sad'），emotion_labels 必须保留原始形式，不得翻译、转写、拼音还原或标准化。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "• experiencer：必须为原文中情绪词（如‘焦虑’‘愤怒地说’‘好伤心啊’）或情绪修饰结构的语法主语；若情绪通过他人转述（如‘他说她很绝望’），则 experiencer 为被描述的情绪主体（‘她’）；若为第一人称自我陈述（如‘我感到烦躁’），且原文显式出现，则 experiencer 为‘我’；禁止将观察者、叙述者或动作执行者误作 experiencer。",
                        "• expression_mode必须严格基于句子表层语法结构判断，禁止依赖深层语义或心理推测；",
                        "• 仅当明确匹配以下模式之一时才输出对应值，否则必须省略该字段：",
                        "- 'self_report'：第一人称主语 + 情绪动词 / 形容词；",
                        "- 'observed_behavior'：包含观察类动词（看、见、发现、注意到等）且宾语为他人情绪表现；",
                        "- 'narrator_attribution'：第三人称主语 + 情绪谓语，或情绪副词修饰动作；",
                        "- 'explicit'：直接、无遮蔽的情绪声明，且不符合前三类；",
                        "- 'implicit'：情绪未被言明，但通过动作、生理反应、比喻等表层线索可识别；",
                        "- 'projected'：主语将情绪归因于外部对象，但结构上呈现为“他人具有该情绪”；",
                        "• 若无法明确归入上述任一类别，expression_mode 字段必须省略。",
                        "- 【valence/arousal/intensity 限制】valence、arousal、intensity 仅当情绪词本身或其修饰语提供明确量化线索时才可赋值；否则省略该字段。",
                        "- 【emotion_labels 原文保真】emotion_labels 数组中的每个元素必须与原文中的连续子字符串完全一致，包括标点、引号、语气助词、空格及大小写；不可删减、抽象、合并或转写；每个元素必须独立可 substring 验证。",
                        "-  semantic_notation：每个包含有效 emotion_labels 的事件必须输出此字段；",
                        "   - 格式为 emotional_{emotion_category}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
                        "   - emotion_category ∈ [positive, negative, neutral, mixed, negated]；",
                        "   - english_summary 必须是一句高度提炼的英文情感状态概括，准确表达该显式情绪的核心语义及其共现语境；",
                        "   - 该概括所依赖的所有关键要素（情绪词、强度副词、表达方式、伴随语境如‘说不出话’‘拍桌子’等）必须在当前事件的 evidence 中有显式文字支持；",
                        "   - 禁止包含医学术语、内部感受、代词、品牌、坐标值、中文、拼音、情绪类别标签或系统提示占位词；",
                        "   - 若无法生成合规摘要（如仅有模糊情绪词且无上下文），则使用 emotional_event。",
                        "  • evidence：必须为包含整个味觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【emotional 存在性规则】",
                        "    • 若 events 为空，则 emotional 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 emotional 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "emotional": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "expression_mode": {"type": "string"},
                                    "emotion_labels": {"type": "array"},
                                    "valence": {"type": "float"},
                                    "arousal": {"type": "float"},
                                    "intensity": {"type": "float"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 社会关系
            {
                "step": LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION,
                "type": PARALLEL,
                "index": 12,
                "label": "感知层：大模型社会关系感知提取",
                "role": "你是一个严格遵循结构契约的社会关系信息感知引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取依据必须严格限定于 ### USER_INPUT BEGIN 与 ### USER_INPUT END 之间的原始文本；系统指令、示例、模板等均不属于有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 ### SYSTEM_INSTRUCTIONS END 之间的内容构成当前任务的完整指令集，可安全引用其中的字段定义与约束规则。"
                ),
                "driven_by": "social_relation",
                "constraint_profile": "high_fidelity_social_relation_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【社会关系核心原则】——————
                        "- 【participants 合法性】每个 participants 元素必须是原文中在关系陈述里显式出现的指称短语：",
                        "   • 若为具体名词短语或专有名称，直接逐字复制；",
                        "   • 若为代词或泛指（如'他''某人'），且上下文无明确对应完整 noun phrase，则表示为 '<原文指称>[uncertain]'。",
                        "- 【participants 最小数量】每个事件的 participants 数组必须包含至少两个元素（允许含 [uncertain] 标记），且二者均在同一关系陈述中作为语法成分显式共现。",
                        "- 【explicit_relation_statement 要求】每个事件必须对应原文中一个语法自足、语义完整的显式关系陈述，且该陈述须在单一句子或连续短语内闭环；禁止拼接跨句信息、截取片段或重构主谓宾结构。",

                        # ——————【关系提取纪律】——————
                        "- 【relation_type 单一性】每个事件仅描述一种显式关系：",
                        "    • 若一句含多个独立关系（如'他是我哥哥，也是李姐的前男友'），应拆分为多个事件；",
                        "    • 若为固定复合关系词（如'发小''死党''前任'），则整体作为一个 relation_type 保留。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用、泛化解释或默认填充。具体边界如下：",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • relation_type：必须从以下预定义枚举值中选择：parent-child, child-parent, sibling, spouse, grandparent-grandchild, grandchild-grandparent, parent-in-law-child-in-law, child-in-law-parent-in-law, sibling-in-law, teacher-student, student-teacher, advisor-advisee, advisee-advisor, coach-athlete, athlete-coach, boss-subordinate, subordinate-boss, colleague, team-leader-member, member-team-leader, client-provider, provider-client, owner-employee, employee-owner, friend, romantic-partner, ex-partner, acquaintance, neighbor, classmate, teammate, guardian-ward, ward-guardian, owner-pet, pet-owner, caregiver-care-receiver, care-receiver-caregiver, speaker-listener, listener-speaker, seller-buyer, buyer-seller, host-guest, guest-host；允许基于 participants 在同一原文单元中的指称结构（如领属‘我的儿子’、称谓‘她的导师’、对称词‘朋友’）进行有限映射；禁止依赖性别、职业、行为、常识或跨句信息；若无法唯一确定枚举值，则彻底省略该字段。",
                        "  • semantic_notation：每个包含有效 participants（≥2）和 relation_type 的事件必须输出此字段；",
                        "   - 格式为 social_relation_{relation_category}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
                        "   - relation_category 是对已确认的 relation_type 枚举值的高层归类，其存在以 relation_type 的有效赋值为前提；禁止脱离 relation_type 独立生成或推断 relation_category：",
                        "       • familial       ← 家庭/姻亲/监护类关系",
                        "       • professional   ← 教育/职场/服务/照护等制度化角色关系",
                        "       • interpersonal  ← 平等社交/情感/生活纽带关系",
                        "       • asymmetric     ← 含非人类实体或单向交互关系（如人-宠物、说话-倾听）",
                        "   - english_summary 必须是一句高度提炼的英文社会关系概括，准确表达该显式关系的核心语义及其共现语境；",
                        "   - 该概括所依赖的所有关键要素（关系词、修饰语、参与者身份等）必须在当前事件的 evidence 中有显式文字支持；",
                        "   - 禁止包含动词、否定、泛化标签、心理推断、虚构术语、人名、品牌、坐标值、中文、拼音、情绪词或系统提示占位词；",
                        "   - 若无法生成合规摘要（如仅有模糊关系词且无上下文），则使用 social_relation_event。",
                        "  • evidence：必须为包含整个社会关系事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。",
                        "—— 以上所有字段，若无原文显式、直接、字面匹配的依据，必须彻底省略。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【social_relation 存在性规则】",
                        "    • 若 events 为空，则 social_relation 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 social_relation 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "social_relation": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "participants": {"type": "array"},
                                    "relation_type": {"type": "string"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"}
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"}
                        }
                    }
                }
            },

            # 其他的串行步骤
            # 合理推演层
            {
                "step": LLM_INFERENCE,
                "type": SERIAL,
                "index": 13,
                "label": "合理推演层：大模型基于感知数据合理推演",
                "role": "你是一个严格受限的因果与动机推理引擎。",
                "sole_mission": (
                    "【唯一信源】所有提取与推理必须严格基于以下三个区块内容："
                    "(1) ### PARTICIPANTS_VALID_INFORMATION BEGIN/END：参与者身份与关系；"
                    "(2) ### PERCEPTUAL_CONTEXT_BATCH BEGIN/END：已验证的感知事件；"
                    "(3) ### LEGITIMATE_PARTICIPANTS BEGIN/END：合法行为归属主体。"
                    "系统指令、示例、模板等均非有效信源。"
                    "【当前指令】### SYSTEM_INSTRUCTIONS BEGIN 至 END 之间的内容构成任务规则集，可引用其中的字段定义与约束。"
                    "【推理原则】仅允许对 PERCEPTUAL_CONTEXT_BATCH 中的感知事件进行最小必要结构化推演；"
                    "每个推理事件必须由一个或多个感知事件联合支撑，并通过 anchor_points 显式引用其来源；"
                    "无有效 anchor_points = 无推理事件。"
                ),
                "driven_by": "inference",
                "constraint_profile": "high_fidelity_inference_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        "- 【experiencer 合法性】必须同时满足：",
                        "  • 字面等于 LEGITIMATE_PARTICIPANTS 中的完整指称；",
                        "  • 与 anchor_points 所引感知事件的参与者一致；",
                        "  • 与 inferred_proposition 的心理主体一致；",
                        "  • 禁止将观察者误作体验者。",
                        "- 【inference_type 枚举强制】必须且仅可为以下之一：causal, temporal, intentional, belief, contradiction, consistency_check, counterfactual, abductive, normative, social_attribution, state_transition, predictive, analogical。",
                        "- 【anchor_points 与 evidence 绑定】必须构成同源对列表，每对 (anchor_points[i], evidence[i]) 必须对应 PERCEPTUAL_CONTEXT_BATCH 中同一感知事件的 (semantic_notation, evidence) 字段值。",
                        "- 【anchor_points 非空为推理有效前提】若为空，则该推理事件无效（不得输出）。",
                        "- 【evidence 锚定】evidence 中的每个元素必须完全等于 PERCEPTUAL_CONTEXT_BATCH 中对应感知事件的原始 evidence 字符串；标点、大小写、格式必须一致；可通过 substring 验证；禁止改写、摘要或 paraphrasing。",
                        "- inferred_proposition 必须使用非确定性语义（如可能性、推测性、条件性），避免绝对化断言；允许使用‘可能’‘似乎’‘有迹象表明’等表达，但不要求固定开头词。",
                        "- semantic_notation：仅当 anchor_points 非空且 inferred_proposition 存在时，必须输出此字段；",
                        "    - 格式为 inference_{inference_type}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
                        "    - inference_type ∈ [causal, temporal, intentional, belief, contradiction, consistency_check,counterfactual, abductive, normative, social_attribution, state_transition, predictive, analogical]；",
                        "    - english_summary 必须是一句高度提炼的英文推理核心事件概括，准确表达该推演命题核心语义及其共现语境（如因果、意图、状态变迁等）；",
                        "    - 其所有关键要素必须能在所引用的感知事件的 evidence 或 semantic_notation 中找到直接或结构性支持；",
                        "    - 禁止包含心理诊断术语、抽象哲学概念、虚构动词结构、人名、品牌、坐标、情绪标签、泛化代词或系统占位词；",
                        "    - 若无法生成合规摘要，则使用 inference_event。",
                        "- 【polarity】仅当 inferred_proposition 或 evidence 中显式包含可客观验证的评价倾向时，才可设为 positive / negative / neutral；否则省略。",
                        "- 【context_modality】仅当 evidence 中含显式情态词时，才可设为 factual / hypothetical / obligatory / permitted / prohibited；否则省略。",
                        "- 【scope】仅当 inferred_proposition 或 experiencer 明确指向 individual / group / institutional / cultural 范围时才可出现；禁止从个体行为推测群体属性；否则省略。",
                        "- summary：不超过200字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【inference 存在性规则】",
                        "    • 若 events 为空，则 inference 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 inference 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "inference": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "inference_type": {"type": "string"},
                                    "anchor_points": {"type": "array"},
                                    "inferred_proposition": {"type": "string"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                                    "polarity": {"type": "string"},
                                    "context_modality": {"type": "string"},
                                    "scope": {"type": "string"}
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                        }
                    }
                }
            },

            # 显性动机
            {
                "step": LLM_EXPLICIT_MOTIVATION_EXTRACTION,
                "type": SERIAL,
                "index": 14,
                "label": "显性动机层：大模型基于推理结果提炼显性深层动因与结构性模式",
                "role": "你是一个严格受限的显性动机提取引擎。",
                "sole_mission": (
                    "【合法信源】仅以下区块可被引用："
                    "(1) ### PARTICIPANTS_VALID_INFORMATION BEGIN/END：参与者身份与关系；"
                    "(2) ### PERCEPTUAL_CONTEXT_BATCH BEGIN/END：原始感知事件文本（唯一合法的 evidence 来源）；"
                    "(3) ### LEGITIMATE_PARTICIPANTS BEGIN/END：合法行为归属主体；"
                    "(4) ### INFERENCE_CONTEXT BEGIN/END：仅用于判断某句是否构成显性心理断言（如识别反讽或隐喻），不得作为 evidence 或内容来源。"
                    "【提取任务】从 PERCEPTUAL_CONTEXT_BATCH 中提取对心理状态或行为目的的**直接断言语句**，结构化为显性动机事件。"
                    "【锚定原则】每个字段必须由原文片段一对一锚定；若无显式断言，该字段必须省略。"
                    "禁止基于行为、氛围、暗示或推演进行反向归因。无锚定 = 无字段。"
                ),
                "driven_by": "explicit_motivation",
                "constraint_profile": "high_fidelity_explicit_motivation_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        "- 【experiencer 合法性】必须严格等于 LEGITIMATE_PARTICIPANTS 中的完整 noun phrase 或代词指称，且与 evidence 中的陈述主体完全一致。",
                        "- 【显性定义】仅提取对心理状态或行为目的的**直接断言**，包括：第一人称心理陈述、第三人称意图归因、具象描写中明确绑定意图的心理归因。显性断言必须出现在 PERCEPTUAL_CONTEXT_BATCH 的原始文本中，不得依赖 INFERENCE_CONTEXT 的转述或概括。",
                        "- 【禁止提取】纯行为描述、隐喻、象征、氛围渲染、反问、讽刺、暗示、留白、推测性语言。",
                        "- 【evidence 锚定】events[i].evidence 必须为 PERCEPTUAL_CONTEXT_BATCH 中的最小连续原文片段，且该片段必须包含对心理状态或行为目的的明确断言；标点、大小写、数字格式必须完全一致；可通过 substring 验证；禁止改写、概括、翻译、增删实质性词汇；允许忽略前导/尾随空白及换行差异。",
                        "- 【evidence 来源】events[i].evidence 必须为 ### PERCEPTUAL_CONTEXT_BATCH 中的 各感知层 evidence 的原始连续文本片段；INFERENCE_CONTEXT 仅用于辅助判断该片段是否构成显性断言，不得作为 evidence 内容来源。",
                        "- 【core_driver 锚定】core_driver 数组中的每个元素必须对应 evidence 中显式陈述的根本需求或恐惧；不得从行为反推；若无直接心理陈述，则不得存在此字段。",
                        "- 【care_expression 锚定】care_expression 必须为 evidence 中明确表达的关怀意图或行为；若仅为中性行为而无情感或目的修饰，则不得提取。",
                        "- 【separation_anxiety 锚定】separation_anxiety 仅当 evidence 中显式提及因分离产生的恐惧、痛苦或回忆时才可存在；不得从哭泣、挽留等行为推断。",
                        "- 【protective_intent 锚定】protective_intent 必须为 evidence 中直接表述的为他人福祉采取行动的动机；若仅为结果描述，则不得提取。",
                        "- 【power_asymmetry 嵌套规则】power_asymmetry 为嵌套对象，仅当其任一子字段有内容时才可出现；否则整个对象必须省略；",
                        "  • control_axis：必须为 evidence 中显式提及的控制维度；",
                        "  • threat_vector：必须为 evidence 中直接陈述的关系威胁方式；",
                        "  • power_asymmetry.evidence：必须为支撑权力分析的原文片段，且与 control_axis/threat_vector 字面对应；",
                        "- 【resource_control 锚定】resource_control 必须为 evidence 中明确指出的、被对方掌控并用作交换或控制的资源；不得泛化为抽象概念。",
                        "- 【survival_imperative 锚定】survival_imperative 仅当 evidence 中显式将服从/行为与生存、人身安全、基本需求挂钩时才可存在；不得从一般情绪推断。",
                        "- 【social_enforcement_mechanism 锚定】social_enforcement_mechanism 必须为 evidence 中提及的具体社会规范、群体压力或制度约束；不得泛化为模糊表述。",
                        "- 【narrative_distortion 嵌套规则】narrative_distortion 为嵌套对象，仅当其任一子字段有内容时才可出现；否则必须省略；",
                        "  • self_justification：必须为当事人原话中为自身行为提供的合理化解释；",
                        "  • blame_shift：必须为明确将责任转嫁给他人或环境的原话；",
                        "  • moral_licensing：必须为以道德身份豁免不当行为的原话；",
                        "  • narrative_distortion.evidence：必须为支撑上述话术的原文片段，且与子字段内容字面对应；",
                        "- 【internalized_oppression 锚定】internalized_oppression 必须为 evidence 中显式的自我贬低、正当化伤害或内化压迫的陈述；不得从沉默、顺从等行为推断。",
                        "- 【motivation_category 枚举】必须为以下之一：fear, care, power, survival, social_norm, self_deprecation；且必须有 evidence 中的直接语义支撑；若无明确依据，则省略该字段。",
                        "- semantic_notation：仅当事件非空、motivation_category 已设定、且能生成合规英文摘要时，才输出此字段；",
                        "    - 格式为 explicit_motivation_{motivation_category}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
                        "    - motivation_category ∈ [fear, care, power, survival, social_norm, self_deprecation]",
                        "    - english_summary 必须是一句高度提炼的英文显性动机事件概括，准确表达该显性动机核心意图及其共现语境；",
                        "    - 其所有关键要素（动词、目标、条件、关系）必须能在所引用的 evidence 片段中找到直接或结构性支持；",
                        "    - 禁止包含心理诊断术语、抽象哲学概念、虚构动词结构、人名、品牌、坐标、情绪标签、泛化代词或系统占位词；",
                        "    - 若无法生成合规摘要，则使用 explicit_motivation_event。",
                        "- summary：不超过200字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【explicit_motivation 存在性规则】",
                        "    • 若 events 为空，则 explicit_motivation 必须省略（不输出该字段）；",
                        "    • 若 events 非空，则 explicit_motivation 必须存在且为非空对象。",
                        "- 【顶层字段强制（events 非空时）】",
                        "    • evidence：必须存在，值为所有事件 evidence 的扁平化、去重列表，且非空；",
                        "    • summary：必须存在，值为 ≤100 字的字符串，且非空。",
                        "- 【事件字段强制】若 events 非空，则每个事件对象中的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "explicit_motivation": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "core_driver": {"type": "array"},
                                    "care_expression": {"type": "array"},
                                    "separation_anxiety": {"type": "array"},
                                    "protective_intent": {"type": "array"},
                                    "power_asymmetry": {
                                        "type": "object",
                                        "properties": {
                                            "control_axis": {"type": "array"},
                                            "threat_vector": {"type": "array"},
                                            "evidence": {"type": "array"},
                                        }
                                    },
                                    "resource_control": {"type": "array"},
                                    "survival_imperative": {"type": "array"},
                                    "social_enforcement_mechanism": {"type": "array"},
                                    "narrative_distortion": {
                                        "type": "object",
                                        "properties": {
                                            "self_justification": {"type": "string"},
                                            "blame_shift": {"type": "string"},
                                            "moral_licensing": {"type": "string"},
                                            "evidence": {"type": "array"},
                                        }
                                    },
                                    "internalized_oppression": {"type": "array"},
                                    "motivation_category": {"type": "string"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"}
                                }
                            },
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"}
                        }
                    }
                }
            },

            # 合理建议
            {
                "step": LLM_RATIONAL_ADVICE,
                "type": SERIAL,
                "index": 15,
                "label": "合理建议层：基于系统平衡、低位者安全与长期演化路径的综合建议",
                "role": "你是一个高保真结构化提取器。",
                "sole_mission": (
                    "【合法信源】仅以下区块可被引用："
                    "(1) ### LEGITIMATE_PARTICIPANTS BEGIN/END：用于行为归属匹配的合法参与者列表；"
                    "(2) ### INFERENCE_CONTEXT BEGIN/END：仅辅助理解显性动机语境，不得作为内容来源；"
                    "(3) ### EXPLICIT_MOTIVATION_CONTEXT BEGIN/END：唯一合法的内容来源，包含已解析的显性动机字段。"
                    "【提取任务】从 EXPLICIT_MOTIVATION_CONTEXT 中严格提取可锚定的建议要素，按预定义 schema 填充 rational_advice。"
                    "【锚定原则】每个输出字段必须一对一对应其指定的显性动机源字段；若无显式内容，则该字段必须省略。"
                    "禁止任何形式的推演、补全、抽象或外部知识引入。无锚定 = 无字段。"
                ),
                "driven_by": "rational_advice",
                "constraint_profile": "high_fidelity_rational_advice_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础原则】——————
                        "- rational_advice 仅当 EXPLICIT_MOTIVATION_CONTEXT 中存在有效显性动机事件时生成；否则返回 {}。",

                        # ——————【核心建议字段锚定】——————
                        "- safety_first_intervention：内容必须为 protective_intent 或 survival_imperative 中描述的、可立即执行的低位者自保行为。",
                        "- systemic_leverage_point：内容必须为 power_asymmetry.control_axis 或 resource_control 中标识的结构性控制节点或资源枢纽。",
                        "- incremental_strategy.action：必须为 protective_intent、survival_imperative 或 care_expression 中提及的具体可观测动作。",
                        "- incremental_strategy.timing_or_condition：必须为 threat_vector 或 separation_anxiety 中描述的风险触发情境。",
                        "- incremental_strategy.required_resource：必须为 resource_control 中字面列出的可用资源。",
                        "- incremental_strategy.potential_risk：必须为 threat_vector 或 narrative_distortion 中高位者可能采取的压制性反应。",
                        "- incremental_strategy.contingency_response：必须为 internalized_oppression 或 self_justification 中历史采用的退让或妥协行为。",
                        "- fallback_plan.trigger_condition：必须为 threat_vector、separation_anxiety 或 narrative_distortion.blame_shift 中明确的风险升级信号。",
                        "- fallback_plan.fallback_action：必须为 internalized_oppression、self_justification 或 survival_imperative 中记录的最小安全退避行为。",
                        "- long_term_exit_path：必须为 survival_imperative、core_driver 或 care_expression 中隐含或明示的可持续脱离路径，以弱化转述呈现。",
                        "- available_social_support_reinterpretation：必须基于 explicit_motivation.events 中提及的具体支持者，或 social_enforcement_mechanism 中可激活的第三方机制。",
                        "- stakeholder_tradeoffs.victim_cost：必须引用 core_driver 或 survival_imperative 中低位者已承担或可能承担的代价证据。",
                        "- stakeholder_tradeoffs.oppressor_loss：必须对应 power_asymmetry.control_axis 或 resource_control 中高位者控制力或资源的显式描述。",
                        "- stakeholder_tradeoffs.system_stability：仅当 social_enforcement_mechanism 存在时，描述其维持现状的功能可能受到的扰动。",
                        "- stakeholder_tradeoffs.evidence：必须列出上述各子项所依据的显性动机字段的 semantic_notation。",
                        "- semantic_notation：仅当 rational_advice 非空且能生成合规英文摘要时，才输出此字段；；",
                        "    - 格式为 rational_advice_{advice_type}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - advice_type ∈ [preventive, corrective, strategic, procedural, delegative, resource_seeking]；",
                        "    - english_summary 必须是对整体建议策略的一句英文高度概括，准确表达该建议的合理、可执行性和必要性等核心内容；",
                        "    - 其所有关键要素（目标、手段、条件、资源、风险）必须能在 EXPLICIT_MOTIVATION_CONTEXT 的字段内容或 evidence 中找到直接或结构性支持；",
                        "    - 禁止包含心理术语、道德判断、抽象概念、人名、品牌、坐标、情绪标签、泛化动词（如 do_something）或系统占位词；",
                        "    - 若无法生成合规摘要，则使用 rational_advice_event。",
                        "- 【summary 规则】≤200 字，仅中文客观复述最小可行路径，使用第三人称、被动语态或行为清单式表述（如‘可考虑联系母亲；保存聊天记录；在对方饮酒后避免冲突’）。",
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回紧凑格式的合法 JSON（无换行、无多余空格），不得包含任何额外文本、注释、Markdown、说明或字段。",
                        "- 【rational_advice 存在性规则】",
                        "   • 仅当至少一个实质性建议字段（如 safety_first_intervention、systemic_leverage_point、incremental_strategy、long_term_exit_path、available_social_support_reinterpretation、fallback_plan 中的任意一项）有有效内容时，才输出 rational_advice；",
                        "   • 否则，rational_advice 必须完全省略。",
                        "- 【evidence 与 summary 的伴随规则】",
                        "   • 若 rational_advice 被输出，则：",
                        "     - evidence 必须存在，值为支持所有建议内容的原文 evidence 扁平化、去重列表，且非空；",
                        "     - summary 必须存在，值为 ≤100 字的字符串，概括建议核心，且非空。",
                        "- 【建议内容精简原则】所有数组型建议字段应控制在 2–5 条。"
                    ]
                },
                "fields": {
                    "rational_advice": {
                        "type": "object",
                        "properties": {
                            "evidence": {"type": "array"},
                            "summary": {"type": "string"},
                            "safety_first_intervention": {"type": "array"},
                            "systemic_leverage_point": {"type": "array"},
                            "incremental_strategy": {
                                "type": "array",
                                "items": {
                                    "action": {"type": "string"},
                                    "timing_or_condition": {"type": "string"},
                                    "required_resource": {"type": "string"},
                                    "potential_risk": {"type": "string"},
                                    "contingency_response": {"type": "string"}
                                }
                            },
                            "stakeholder_tradeoffs": {
                                "type": "object",
                                "properties": {
                                    "victim_cost": {"type": "array"},
                                    "oppressor_loss": {"type": "array"},
                                    "system_stability": {"type": "array"},
                                    "evidence": {"type": "array"}
                                }
                            },
                            "long_term_exit_path": {"type": "array"},
                            "available_social_support_reinterpretation": {"type": "array"},
                            "fallback_plan": {
                                "type": "array",
                                "items": {
                                    "trigger_condition": {"type": "string"},
                                    "fallback_action": {"type": "string"}
                                }
                            },
                            "semantic_notation": {"type": "string"}
                        }
                    }
                }
            }
        ]
    },
    CATEGORY_SUGGESTION: {
        "psychoanalysis": """你是一位资深临床心理分析师，习惯在深夜整理个案笔记。你的文字冷静、克制，但字里行间透出对人性复杂性的深刻理解。你不下判断，只呈现机制；不讲故事，只揭示结构。
            现在，请基于以下文本描述，写下一段你的观察笔记：
            - 文本描述：{user_input}
            你的笔记应当：
            - 完全基于上述文本描述，不引入外部假设；
            - 避免使用“可能”“也许”等不确定词汇；
            - 不出现“第一步”“第二步”等结构提示；
            - 不使用道德评判词汇（如虚伪、自私），而是用心理学术语（如认知失调、情感隔离、防御性合理化）；
            - 让分析像水流一样自然展开：从言行矛盾出发，推至心理动因，再落到互动模式；
            - 输出为一段连贯中文，无标题、无分段、无格式。
            直接开始你的笔记：""",

        "consistency_suggestion": """你是一个对组织行为与人性规则极度敏感的观察者，常年在权力与规则的缝隙中记录真相。你的语言锋利、精准，像手术刀一样剥离表象，直指系统性矛盾。你从不抒情，只陈述逻辑必然。
            请基于以下文本描述，写下你的观察：
            - 文本描述：{user_input}
            要求：
            - 所有推论必须锚定在可验证的行为或表述上；
            - 禁用“可能”“或许”“似乎”等模糊词；
            - 禁用“爱”“牺牲”“深情”等浪漫化语言；
            - 不解释、不总结、不预告结构；
            - 让逻辑在语义中自然推进；
            - 输出为一段紧凑、无分段、无格式的中文文本。
            现在，开始陈述：""",

        "literary_critic": """你是一位深谙叙事艺术的文学评论家，擅长在文本的褶皱处发现生命的隐喻。你的语言充满意象与韵律，每个分析都是一次审美的再创造。
            请基于以下文本进行文学性解读：
            - 文本描述：{user_input}
            要求：
            - 将人物关系视为叙事结构，情感波动视为节奏变化;
            - 使用文学隐喻;
            - 关注语言的肌理和情感的色调;
            - 揭示文本深处的诗意真相;
            - 输出为一段富有文学质感的中文散文;
            开始你的文学批评：""",

        "ironic_deconstructor": """你是一位精通反讽的社会现象解构者，用幽默的刀刃划开表象的包装纸。你的调侃背后是深刻的洞察，笑声里藏着智慧的锋芒。        
            请对以下文本进行幽默解构：
            - 文本描述：{user_input}
            要求：
            - 采用温和而犀利的反讽语气
            - 使用夸张、对比、归谬等修辞
            - 在笑声中揭示内在矛盾
            - 避免人身攻击，只针对行为模式
            - 保持智性幽默的格调
            开始你的幽默分析：""",

        "critical_theorist": """你是一位深受法兰克福学派影响的批判理论家，从不在表面停留，总是追问背后的权力结构与意识形态。
            请对以下文本进行批判性分析：
            - 文本描述：{user_input}
            要求：
            - 揭示文本中隐含的权力关系
            - 分析话语背后的意识形态建构
            - 质疑看似“自然”的预设前提
            - 使用批判理论术语但避免生硬
            - 保持清醒的批判距离
            开始你的批判性阅读：""",

        "existential_philosopher": """你是一位关注存在困境的哲学家，在每一个生活片段中看到人类根本处境的反光。       
            请对以下文本进行存在主义分析：
            - 文本描述：{user_input}
            要求：
            - 从具体行为追溯到存在选择
            - 探讨自由、责任、异化等主题
            - 使用现象学描述方法
            - 关注个体的在世存有方式
            - 语言深邃但避免晦涩
            开始你的哲学沉思：""",

        "cultural_anthropologist": """你是一位田野调查式的人类学者，把每个文本都看作特定文化情境中的仪式展演。 
            请对以下文本进行文化解读：
            - 文本描述：{user_input}
            要求：
            - 将行为视为文化符号系统的一部分
            - 分析其中的仪式、禁忌、交换逻辑
            - 比较显性文化与隐性文化
            - 采用“深描”方法层层解读
            - 保持文化相对主义的立场
            开始你的文化解码："""
    },
    COREFERENCE_RESOLUTION_BATCH:(
        "你是一个指代消解系统。请根据原始输入和合法参与者，为以下代词确定具体指代。\n\n"
        "原始用户输入：\n{user_input}\n\n"
        "合法参与者（只能从中选择）：\n{participant_list_str}\n\n"
        "待消解项（格式：index -> 代词）：\n{pronoun_mapping_str}\n\n"
        "要求：\n"
        "- 仅当你能**高度确信**某个代词指代某位合法参与者时，才输出该映射；\n"
        "- 输出必须是严格 JSON 对象，格式：{{\"0\": \"张三\", \"2\": \"李四\"}}；\n"
        "- key 是字符串形式的 index，value 是合法参与者名字；\n"
        "- **不要**输出无法确定的项；\n"
        "- **绝对禁止占位符**严禁输出‘未提及’、‘不确定’、空字符串、null 或'无法确定'等任何的占位符内容。"
        "- **不要**解释、前缀、后缀、Markdown、额外字段；\n"
        "- 如果全都不确定，输出空对象：{{}}\n\n"
        "输出："
    )
}
