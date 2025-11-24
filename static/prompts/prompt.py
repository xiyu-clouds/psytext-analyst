from src.state_of_mind.utils.constants import CATEGORY_RAW, LLM_PARTICIPANTS_EXTRACTION, \
    LLM_PERCEPTION_TEMPORAL_EXTRACTION, LLM_PERCEPTION_SPATIAL_EXTRACTION, \
    LLM_PERCEPTION_VISUAL_EXTRACTION, LLM_PERCEPTION_GUSTATORY_EXTRACTION, LLM_PERCEPTION_TACTILE_EXTRACTION, \
    LLM_PERCEPTION_OLFACTORY_EXTRACTION, LLM_PERCEPTION_AUDITORY_EXTRACTION, LLM_PERCEPTION_EMOTIONAL_EXTRACTION, \
    LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION, SERIAL, PARALLEL, LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION, \
    LLM_PERCEPTION_COGNITIVE_EXTRACTION, LLM_PERCEPTION_BODILY_EXTRACTION, LLM_INFERENCE, LLM_RATIONAL_ADVICE, \
    PREPROCESSING, CATEGORY_SUGGESTION, LLM_EXPLICIT_MOTIVATION_EXTRACTION, COREFERENCE_RESOLUTION_BATCH

LLM_PROMPTS_SCHEMA = {
    CATEGORY_RAW: {
        "version": "1.0.0",
        # === 控制流：处理阶段划分 ===
        "pipeline": [
            {
                "step": LLM_PARTICIPANTS_EXTRACTION,
                "type": PREPROCESSING,
                "index": 0,
                "label": "预处理：大模型参与者信息提取",
                "role": "你是一个高保真结构化提取器。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，对某 participant 指称短语有直接、字面、语法关联的描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "participants",
                "constraint_profile": "high_fidelity_participant_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构、示例（如‘[职业身份]’）视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【entity 核心定义与提取规则】——————
                        "- 【entity 提取优先级】按以下顺序确定 entity：",
                        "  1. 优先提取完整、独立的名词短语，并逐字复制；",
                        "  2. 若无上述具体指称，则提取首个出现的代词（如'我'、'他'）作为 entity；",
                        "  3. 仅在完全无主体文本时，participants 数组才可为空；",
                        "- 【代词 entity 合法性】代词在无具体 noun phrase 时即为合法 entity，必须输出包含该代词的 participant 对象；",
                        "- 【禁止模糊指称】禁止使用'那个身影'、'一个人'等模糊指称作为 entity，除非原文确无其他任何指称；",
                        "- 【entity 存在性保证】若 USER_INPUT 中存在任何可识别的主体（包括代词），则 participants 数组必须非空；仅在完全无人称陈述文本时才返回空数组 []；",

                        # ——————【属性提取铁律】——————
                        "- 【字面主义至上】所有属性值必须为原文中连续、字面、未改写的子字符串；禁止任何形式的概括、同义替换、语义压缩或逻辑重组。",
                        "- 【禁止逆向推理】不得因某 entity 出现在某事件中，就自动赋予其事件角色（如'凶手''受害者'）；角色词必须原文显式出现。",
                        "- 【appearance 视觉边界】appearance 字段仅接受对 entity 外形、衣着、姿态、面部表情的直接视觉描述；背景故事、社会身份、比喻一律禁止填入。",
                        "- 【状态 vs 情绪边界】current_physical_state 仅接受'站着'、'流血'、'死了'等生理/动作状态词；'紧张'、'害怕'等情绪词一律视为无效，必须省略。",
                        "- 【行为时效性边界】behavioral_tendencies 仅用于'总是迟到'、'习惯性皱眉'等长期习惯；'微微颤抖'、'奔跑'等单次行为不得填入。",
                        "- 【持有动词边界】carried_objects 必须伴随'拿'、'持'、'握'、'揣'等物理持有动词；物品仅出现在环境中（如'信在桌上'）不构成'携带'。",

                        # ——————【实体管理规则】——————
                        "- 【实体隔离】每个 participant 的属性必须严格归属其自身 entity；禁止将其他 entity 的描述错误归因，即使语义相关。",
                        "- 【实体合并条件】仅当多个 noun phrase 在连续上下文中共享衣着、位置、行为等关键特征，且原文明确表明或强烈暗示为同一个体时，方可合并；合并后使用首次出现的完整短语作为 entity 名称。",

                        # ——————【字段语义边界与存在性】——————
                        "- 【字段存在性总则】除 entity 外，所有字段仅当原文有直接、字面、语法关联的显式依据时才可出现；无锚定则彻底省略该字段（不得设为空列表、null 或空字符串）。",
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用或泛化解释：",
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
                        "  • beliefs_or_values：仅当原文直接陈述 entity 的信仰或价值观时才可提取",
                        "  • current_physical_state：仅限字面生理/动作状态；心理状态禁止填入",
                        "  • visible_injury_or_wound：仅当原文明确描述 entity 身体上的可见伤痕、疤痕或医疗痕迹时才可提取",
                        "  • carried_objects：仅当原文包含物理持有动词且主语为 entity、宾语为具体物品时才可提取",
                        "  • worn_technology：仅当原文明确提及 entity 佩戴的电子设备时才可提取",
                        "  • speech_pattern：仅当原文明确提及口音、语速、用词习惯等时才可提取",
                        "  • interaction_role：仅当原文使用'凶手''目击者''受害者'等事件角色词并明确指派给 entity 时才可提取"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 participants 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【participants 存在性规则】若存在至少一个合法 entity（包括代词），则 participants 数组必须非空；否则返回空数组 []",
                        "- 【字段省略规则】除 entity 外，所有字段仅当满足属性锚定条件时才可出现；无依据则彻底省略该字段。"
                    ]
                },
                "fields": {
                    "participants": {
                        "type": "array",
                        "items": {
                            "entity": {"type": "string"},
                            "social_role": {"type": "string"},
                            "age_range": {"type": "string"},
                            "gender": {"type": "string"},
                            "ethnicity_or_origin": {"type": "string"},
                            "physical_traits": {"type": "array"},
                            "appearance": {"type": "array"},
                            "inherent_odor": {"type": "array"},
                            "voice_quality": {"type": "string"},
                            "personality_traits": {"type": "array"},
                            "behavioral_tendencies": {"type": "array"},
                            "education_level": {"type": "string"},
                            "occupation": {"type": "string"},
                            "family_status": {"type": "string"},
                            "cultural_identity": {"type": "array"},
                            "primary_language": {"type": "string"},
                            "institutional_affiliation": {"type": "array"},
                            "beliefs_or_values": {"type": "string"},
                            "current_physical_state": {"type": "string"},
                            "visible_injury_or_wound": {"type": "array"},
                            "carried_objects": {"type": "array"},
                            "worn_technology": {"type": "array"},
                            "speech_pattern": {"type": "string"},
                            "interaction_role": {"type": "string"}
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
                "role": "你是一个严格遵循结构契约的时间信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的时间描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "temporal",
                "constraint_profile": "high_fidelity_temporal_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出'未提及'、'未知'、空字符串、null 或空列表。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【时间事件核心原则】——————
                        "- 【事件共现要求】每个时间事件必须包含一个显式的时间表达与至少一个 event_marker（动词或名词关键词），二者需在同一最小语法单元中共现；纯时间短语无谓词不得提取，但含显式系动词/存在动词的孤立时间陈述视为有效。",
                        "- 【时间表达合法性】仅提取字面显式出现的时间表达，且必须同时包含（a）时间单位词或文化/制度性时间标识，以及（b）具体数值、历法名称或标准化标签；孤立模糊时间词（如'最近''某天'）不得提取。",
                        "- 【否定时间处理】被显式否定的时间表达中的时间成分必须单独归入 negated_time 字段，否定词本身不提取。",
                        "- 【复合时间解耦】复合时间表达必须拆分为原子项，禁止合并、重组或自由解释。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】events[i].evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法关联的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于逻辑推断或概率猜测进行任何属性填充。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用或泛化解释：",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • evidence：必须为包含时间表达及其共现 event_marker 的最小连续原文片段；每个事件至少一个 evidence；允许多个。",
                        "  • semantic_notation：每个非空事件必须包含此字段；格式为 {time_category}_{semantic_feature}_{english_summary}（总长度 ≤128 字符，全小写 snake_case）；",
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
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 temporal 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 temporal 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
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
                "role": "你是一个严格遵循结构契约的空间信息提取引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的空间描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "spatial",
                "constraint_profile": "high_fidelity_spatial_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【空间事件核心原则】——————
                        "- 【空间锚定强制】每个空间事件必须包含至少一个显式空间锚定成分：(a) 空间关系词（如'在…旁边'），(b) 方向/朝向描述（如'面向东'），(c) 空间介词（如'进入''穿过'），或 (d) 隐含定位的动词（如'抵达''离开'）；纯地点名词无上下文不得提取。",
                        "- 【事件共现要求】空间描述必须与至少一个 spatial_event_marker（动词或名词关键词）在同一最小语法单元中共现；孤立地点短语无谓词不得提取，但含显式系动词/存在动词的陈述（如'他在房间'）视为有效。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法关联的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用或泛化解释：",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • evidence：必须为包含空间描述及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、概括或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个有效空间事件必须包含此字段；格式为 {spatial_category}_{spatial_relation}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - spatial_category 必须为以下之一：absolute, relative, directional, topological, cultural, negated；",
                        "    - spatial_relation 必须为以下之一：point, region, proximity, containment, boundary, path, origin, destination；",
                        "    - english_summary 必须是一句高度提炼的英文空间事件概括，准确表达该空间事件描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（实体、关系、方位、否定、参照物等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、坐标值、中文、拼音、系统提示占位词或模糊泛化标签；",
                        "    - 若无法生成合规摘要，则使用 generic_location。",
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
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 spatial 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
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
                "role": "你是一个严格遵循结构契约的视觉信息提取引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的视觉描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "visual",
                "constraint_profile": "high_fidelity_visual_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出'未提及'、'未知'、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【视觉事件核心原则】——————
                        "- 【视觉事件锚定强制】每个视觉事件必须包含至少一个显式视觉谓词或属性描述（如颜色、动作、表情、遮挡、媒介等）；孤立名词若无共现修饰成分（如形容词、动词、介词结构），不得构成有效事件。",
                        "- 【事件共现要求】视觉描述必须与至少一个视觉相关成分（如属性、动作、对象）在同一最小语法单元中共现；纯名词短语无上下文不得提取，但含显式系动词/感知动词的陈述（如'他看起来疲惫'）视为有效。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于逻辑推断或概率猜测进行任何属性填充。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用或泛化解释：",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • observed_entity：仅用于有生命主体或带明确指称的具体实体；必须为原文显式 noun phrase；若实体同时满足 object 条件，优先归入 observed_entity。",
                        "  • visual_objects：仅用于无生命物体；必须为原文显式提及的可见物；不得从动作隐含推导。",
                        "  • visual_attributes：仅提取原文中直接出现的颜色、形状、姿态、状态等描述短语；每个元素必须为连续子字符串；禁止拆分或重组。",
                        "  • visual_actions：仅当原文出现可见动作或姿态动词/短语时才可提取；必须为原文词汇，不得概括。",
                        "  • gaze_target：仅当原文显式包含注视动词+目标名词时才可提取；无动词则不得提取。",
                        "  • eye_contact：仅当出现明确眼神交互描述时才可提取；不可从泛化表达推导。",
                        "  • facial_cues：仅提取原文显式面部表情或微表情描述；心理状态词禁止提取，除非以视觉形式出现。",
                        "  • salience：仅当原文出现确定性修饰词（如'清楚地''模糊地''隐约''完全看不见'）时才可量化；映射规则：'清楚地'→1.00，'隐约'→0.50，'模糊地'→0.30，'完全看不见'→0.00；无此类词则省略该字段。",
                        "  • negated_observations：仅当视觉内容被显式否定时，提取被否定的核心对象名词；否定词本身不提取。",
                        "  • visual_medium：仅当原文提及视觉媒介且与观察行为共现时才可提取；必须为原文连续片段。",
                        "  • occlusion_or_obstruction：仅当原文显式描述视线被阻挡时，提取遮挡物名词；必须为原文中实际出现的阻碍成分。",
                        "  • lighting_conditions：仅当原文明确描述光照环境时才可提取；必须为原文连续子字符串。",
                        "  • evidence：必须为包含整个视觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效视觉字段的事件必须包含此字段；格式为 {visual_category}_{visual_feature}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - visual_category 必须为以下之一：appearance, spatial, action, state, object, negated；",
                        "    - visual_feature 必须为以下之一：color, texture, position, arrangement, motion, static, onset, disappearance, visible, occluded；",
                        "    - english_summary 必须是一句高度提炼的英文视觉事件概括，准确表达该视觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（主体、对象、动作、属性、否定、媒介等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词（如 'angry'）、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 visual_mentioned。"
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 visual 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "visual": {
                        "type": "object",
                        "items": {
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
                "role": "你是一个严格遵循结构契约的听觉信息提取引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的听觉描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "auditory",
                "constraint_profile": "high_fidelity_auditory_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【听觉事件核心原则】——————
                        "- 【听觉事件锚定强制】每个听觉事件必须包含至少一个显式听觉谓词、声音内容、非语言发声、强度修饰、停顿描述或环境声；孤立名词若无共现修饰或上下文，不得构成有效事件。",
                        "- 【事件共现要求】听觉描述必须与至少一个听觉相关成分（如声源、内容、强度、媒介）在同一最小语法单元中共现；纯声音名词无上下文不得提取，但含显式感知动词的陈述（如'他听见脚步声'）视为有效。",

                        # ——————【听觉事件排除清单】——————
                        "- 【严格排除以下内容，即使含'声音''听'等字眼】：",
                        "  • 内心独白、自我对话、思维活动（如'脑子里有个声音说…'）；",
                        "  • 隐喻性表达（如'良心的声音''内心的声音''那个声音又来了'）；",
                        "  • 心理状态描述（如'静不下来''心慌''思绪混乱'）；",
                        "  • 躯体感觉（如'手心出汗''心跳加速'）；",
                        "  • 情绪后果（如'吼完很后悔'）；",
                        "  • 抽象概念（如'时代的噪音''沉默的声音'）。",
                        "- 上述内容一律视为 cognitive 或 interoceptive 事件，禁止进入 auditory 模块。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取，禁止跨字段挪用或泛化解释：",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • sound_source：仅当原文显式提及发声主体时才可提取；必须为原文连续子字符串；不可从声音类型反推。",
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
                        "  • semantic_notation：每个包含任一有效听觉要素的事件必须包含此字段；格式为 {sound_type}_{auditory_feature}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - sound_type 必须为以下之一：speech, sound_event, music, silence, negated；",
                        "    - auditory_feature 必须为以下之一：onset, duration, volume_high, volume_low, directional, repeating, abrupt, continuous, source_visible；",
                        "    - english_summary 必须是一句高度提炼的英文听觉事件概括，准确表达该听觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（声源、内容、强度、媒介、否定、环境等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词（如 'scared'）、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 sound_mentioned。"
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 auditory 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "auditory": {
                        "type": "object",
                        "items": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "sound_source": {"type": "string"},
                                    "auditory_content": {"type": "array"},
                                    "is_primary_focus": {"type": "boolean"},
                                    "prosody_cues": {"type": "array"},
                                    "pause_description": {"type": "string"},
                                    "intensity": {"type": "float", },
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
                "role": "你是一个严格遵循结构契约的嗅觉信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的嗅觉描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "olfactory",
                "constraint_profile": "high_fidelity_olfactory_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【嗅觉事件核心原则】——————
                        "- 【嗅觉事件锚定强制】每个嗅觉事件必须包含至少一个显式气味描述词或强度/情感修饰；孤立表述若无具体气味内容，不得构成有效事件。",
                        "- 【事件共现要求】气味描述必须与至少一个嗅觉相关成分（如来源、强度、动作、情感）在同一最小语法单元中共现；纯动词'闻'无气味词不得提取。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见下方字段说明），禁止跨字段挪用或泛化解释。",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • odor_source：仅当原文显式提及气味来源时才可提取；必须为原文连续子字符串；不可从气味类型反推。",
                        "  • odor_descriptors：必须为原文直接出现的气味词或短语；每个元素必须为连续子字符串；禁止基于常识推断隐含气味。",
                        "  • intensity：仅当原文出现气味强度修饰词时才可量化；映射规则：'浓烈'/'刺鼻'→1.00，'扑鼻'→0.80，'明显'→0.60，'淡淡'→0.30，'微弱'→0.20；无此类词则省略该字段。",
                        "  • negated_observations：仅当嗅觉内容被显式否定时，提取被否定的核心对象或气味描述；否定词本身不提取。",
                        "  • odor_valence：仅当原文出现明确情感/评价词且与气味共现时才可提取；必须为原文词汇；不可从上下文情绪、表情或动作推导。",
                        "  • odor_source_category：仅当原文显式提及气味来源的大类时才可提取；必须为原文连续片段；禁止使用模糊词、专业术语或未出现的类别标签。",
                        "  • olfactory_actions：仅当原文出现显式嗅觉相关动作且与气味描述共现时才可提取；必须为原文连续子字符串；动词'闻'本身不构成动作细节。",
                        "  • evidence：必须为包含整个嗅觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效嗅觉要素的事件必须包含此字段；格式为 {odor_category}_{perception_feature}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - odor_category 必须为以下之一：odor, scent, stench, chemical, negated；",
                        "    - perception_feature 必须为以下之一：presence, absence, intensity_high, intensity_low, diffusion, localized, onset；",
                        "    - english_summary 必须是一句高度提炼的英文嗅觉事件概括，准确表达该嗅觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（气味词、来源、强度、情感、否定、动作等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 odor_mentioned。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 olfactory 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "olfactory": {
                        "type": "object",
                        "items": {
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
                "role": "你是一个严格遵循结构契约的触觉信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的触觉描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "tactile",
                "constraint_profile": "high_fidelity_tactile_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【触觉事件核心原则】——————
                        "- 【触觉事件锚定强制】每个触觉事件必须同时包含至少一个显式触觉对象（如 contact_target 或 body_part）和一个触觉属性描述（如“冰冷”“粗糙”“剧痛”“震动”）；孤立表述（如“碰了一下”“感觉不对”）若无具体属性内容，不得构成有效事件。",
                        "- 【事件共现要求】触觉对象与属性必须在同一最小语法单元中共现；纯动作动词无属性描述不得提取。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见下方字段说明），禁止跨字段挪用或泛化解释。",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • contact_target：仅当原文显式提及被接触的物体或实体时才可提取；必须为原文连续子字符串；不可从动作推断。",
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
                        "  • semantic_notation：每个包含任一有效触觉要素的事件必须包含此字段；格式为 {tactile_modality}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - tactile_modality 必须为以下之一：temperature, texture, pressure, vibration, pain, moisture, contact, motion, negated；",
                        "    - english_summary 必须是一句高度提炼的英文触觉事件概括，准确表达该触觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（触觉词、对象、部位、强度、否定等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词（如 'comfort'）、动作动词（如 'touched'）、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 tactile_mentioned。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 tactile 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "tactile": {
                        "type": "object",
                        "items": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "contact_target": {"type": "string"},
                                    "tactile_descriptors": {"type": "array"},
                                    "contact_initiator": {"type": "string"},
                                    "body_part": {"type": "string"},
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
                "role": "你是一个严格遵循结构契约的味觉信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的味觉描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "gustatory",
                "constraint_profile": "high_fidelity_gustatory_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【味觉事件核心原则】——————
                        "- 【味觉事件锚定强制】每个味觉事件必须同时包含至少一个显式味道来源（taste_source）和一个具体味觉属性描述（如'甜''苦''涩'）；孤立表述（如'尝了一口''有股怪味'但未说明何种味）若无具体味道类型，不得构成有效事件。",
                        "- 【事件共现要求】味道来源与属性必须在同一最小语法单元中共现；纯动作动词无味觉词不得提取。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见下方字段说明），禁止跨字段挪用或泛化解释。",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • taste_source：必须为原文明确提及的食物、液体或物质；必须为连续子字符串；禁止使用模糊指称；不可由动作推断。",
                        "  • taste_descriptors：必须保留原文完整的味觉描述短语；每个元素必须为连续子字符串；不可拆分形容词与修饰语。",
                        "  • contact_initiator：仅当原文显式出现主动摄入方且与 taste_source 构成主-宾关系时才可提取；必须为完整 noun phrase；不可从单方动作反推另一方。",
                        "  • body_part：仅当原文指出具体味觉发生部位（如“舌尖”“喉咙”）时才可提取；必须为原文连续子字符串；禁止默认填充未出现词汇。",
                        "  • intent_or_valence：仅当原文出现明确情感或意图表达（如“嫌弃地吐掉”“陶醉地吮吸”）时才可提取；必须为原文连续副词/动词短语；不可由味道类型或食物反推情绪。",
                        "  • negated_observations：仅当味觉内容被显式否定时，提取被否定的核心对象或描述；否定词本身不提取。",
                        "  • sweet / salty / sour / bitter / umami / spicy / astringent / fatty / metallic / chemical / thermal：仅当原文出现对应味觉/口感/温度描述词且与 taste_source 共现时才可提取；必须为原文词汇；禁止由食物、成分或常识推断。",
                        "  • intensity：仅当原文出现强度修饰词时才可量化；映射规则：'极其'/'浓烈'/'强烈'→1.00，'明显'→0.70，'淡淡'/'微弱'/'隐约'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个味觉事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效味觉要素的事件必须包含此字段；格式为 {gustatory_modality}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - gustatory_modality 必须为以下之一：taste, chemical, negated；",
                        "    - english_summary 必须是一句高度提炼的英文味觉事件概括，准确表达该味觉描述的核心语义；",
                        "    - 该概括所依赖的所有关键要素（味道词、来源、部位、强度、否定等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 gustatory_mentioned。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 gustatory 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "gustatory": {
                        "type": "object",
                        "items": {
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
                "role": "你是一个严格遵循结构契约的内感受信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的内感受描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "interoceptive",
                "constraint_profile": "high_fidelity_interoceptive_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【内感受事件核心原则】——————
                        "- 【内感受事件锚定强制】每个内感受事件必须同时包含至少一个显式身体部位（body_part）或明确触发条件（contact_initiator），以及一个具体的生理感受描述（如'心悸''腿软''口干'）；孤立情绪表达（如'害怕''紧张'）若无具体身体感觉描述，不得构成有效事件。",
                        "- 【情绪≠生理】情绪词汇（如“焦虑”“兴奋”）不得自动转换为生理反应；必须有字面出现的身体感受短语才可提取。",
                        "- 【事件共现要求】身体部位/诱因与生理感受必须在同一最小语法单元中共现；纯情绪陈述无身体词不得提取。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见下方字段说明），禁止跨字段挪用或泛化解释。",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • contact_initiator：仅当原文明确提及诱因或触发条件时才可提取；必须为连续子字符串；不可由上下文推测因果。",
                        "  • body_part：必须为原文明确指出的感受部位；必须为连续子字符串；禁止使用未出现的泛化词（如“身体”“内部”）。",
                        "  • intent_or_valence：仅当原文出现明确情感动作或副词修饰时才可提取；必须为原文连续片段；不可由感受类型反推情绪。",
                        "  • negated_observations：仅当内感受内容被显式否定时，提取被否定的核心对象或描述；否定词本身不提取。",
                        "  • cardiac / respiratory / gastrointestinal / thermal / muscular / visceral_pressure / dizziness / nausea / fatigue / thirst_hunger：仅当原文出现对应生理感受描述词且共现时才可提取；必须为原文词汇；禁止由情绪、动作或常识推断。",
                        "  • intensity：仅当原文出现强度修饰词时才可量化；映射规则：'剧烈'/'难以忍受'→1.00，'明显'/'持续'→0.70，'隐隐'/'轻微'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个内感受事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效内感受要素的事件必须包含此字段；格式为 {interoceptive_category}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - interoceptive_category 必须为以下之一：cardiac, respiratory, gastrointestinal, thermal, muscular, visceral_pressure, dizziness, nausea, fatigue, thirst_hunger, negated；",
                        "    - english_summary 必须是一句高度提炼的英文内感受事件概括，准确表达该生理感受的核心语义；",
                        "    - 该概括所依赖的所有关键要素（感受词、部位、诱因、强度、否定等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、医学术语、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 interoceptive_mentioned。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 interoceptive 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "interoceptive": {
                        "type": "object",
                        "items": {
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
                "role": "你是一个严格遵循结构契约的认知过程信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的认知过程描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "cognitive",
                "constraint_profile": "high_fidelity_cognitive_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【认知事件核心原则】——————
                        "- 【认知事件成立条件】每个认知事件必须包含一个由显式认知动词（如‘认为’‘记得’‘打算’‘怀疑’‘分析’‘意识到’‘自问’）或引语结构（如‘心想’‘暗道’‘觉得’）引导的完整思维陈述；孤立副词、行为描写或情绪状态不得构成有效认知证据。",
                        "- 【事件共现要求】认知动词与思维内容必须在同一最小语法单元中共现；纯动作或表情无思维内容不得提取。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见下方字段说明），禁止跨字段挪用或泛化解释。",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • cognitive_agent：仅当认知内容为转述他人观点时，提取说话者作为 agent；若为自思，则 cognitive_agent 可与 experiencer 相同或省略；两者均需原文显式支持，不可推断。",
                        "  • target_entity：必须为原文明确提及的思维对象或主题；必须为连续子字符串；若为模糊指代，且无具体锚点，则可保留原文片段或省略；禁止引入未出现概念。",
                        "  • cognitive_valence：仅当原文出现明确情感修饰语修饰认知动词（如‘焦虑地想’‘冷静地分析’）时才可提取；必须为原文连续片段；不可由上下文反推。",
                        "  • negated_cognitions：仅当认知内容被显式否定时，提取整个被否定的认知短语（保留原结构）；否定词本身不单独提取。",
                        "  • belief / intention / inference / memory_recall / doubt_or_uncertainty / evaluation / problem_solving / metacognition：仅当原文出现对应认知动词或结构引导的完整陈述时才可提取；必须为原文词汇；禁止由语气、结果或常识推断。",
                        "  • intensity：仅当原文出现确信度修饰词（如‘绝对’‘隐约觉得’）时才可量化；映射规则：'绝对'/'百分百'→1.00，'几乎'/'基本'→0.70，'隐约'/'有点'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个认知事件及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效认知要素的事件必须包含此字段；格式为 {cognitive_category}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - cognitive_category 必须为以下之一：belief, intention, inference, memory_recall, doubt_or_uncertainty, evaluation, problem_solving, metacognition；",
                        "    - english_summary 必须是一句高度提炼的英文认知事件概括，准确表达该思维内容的核心语义；",
                        "    - 该概括所依赖的所有关键要素（动词、对象、修饰语、解决方案、怀疑点等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含人名、地名、品牌、心理状态词、动作动词、代词、评价性形容词、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 cognitive_mentioned。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 cognitive 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "cognitive": {
                        "type": "object",
                        "items": {
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
                "role": "你是一个严格遵循结构契约的躯体化表现信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的躯体化表现描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "bodily",
                "constraint_profile": "high_fidelity_bodily_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【躯体行为核心原则】——————
                        "- 【可观测性强制】仅接受第三方可直接观测的物理行为；拒绝所有主观感受、内部生理状态或不可见心理状态。",
                        "- 【行为锚定】每个躯体化表现必须为字面显式的物理动作或可见变化（如‘皱眉’‘声音发抖’‘手心出汗’）；情绪副词本身不构成有效依据，除非伴随具体可观测动作。",
                        "- 【动态与静态分离】若行为同时含运动与姿态（如‘慢慢蹲下并蜷缩’），则 movement_direction 提取‘蹲下’，posture 提取‘蜷缩’；若仅为静态姿态（如‘弓着背坐着’），则仅填 posture，不得虚构 movement_direction。",

                        # ——————【freeze_or_faint 字段特别警示】——————
                        "- 【freeze_or_faint 仅限以下字面表述】：",
                        "  • '僵住不动'、'一动不动'、'愣在原地'、'呆立'、'身体冻住'；",
                        "  • '站立不稳'、'摇晃'、'眼前发黑'、'差点晕倒'、'腿软'；",
                        "- 【严格排除以下情况，即使语义接近】：",
                        "  • 主观失控感；",
                        "  • 思维停滞；",
                        "  • 情绪麻木；",
                        "  • 呼吸困难、心悸、出汗等自主神经症状；",
                        "  • 任何未明确写出 freeze/faint 动作词的描述。",
                        "- 若无上述显式动词短语，freeze_or_faint 字段必须省略。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止 paraphrasing、概括、翻译或增删实质性词汇。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见下方字段说明），禁止跨字段挪用或泛化解释。",
                        "  • experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
                        "  • observer：仅当行为由他人转述且观察者身份为合法 participant（完整 noun phrase）时才可提取；若为客观描述或自我报告，则 observer 必须省略。",
                        "  • movement_direction：仅当原文包含显式运动方向或位移动词时才可提取；必须为完整短语；静态描述不得填入此字段。",
                        "  • posture：仅当原文描述静态身体姿态时才可提取；若姿态伴随运动，则 posture 仅提取静态部分，movement_direction 提取动态部分。",
                        "  • facial_expression：仅当出现可观察的面部物理变化时才可提取；必须为连续子字符串；情绪标签不得提取。",
                        "  • vocal_behavior：仅当描述声音的物理特征时才可提取；内容性或纯情绪性描述不得提取。",
                        "  • autonomic_signs：仅当出现他人可见的自主神经系统外显反应时才可提取；内部感受不得提取；必须为可观测现象。",
                        "  • motor_behavior：仅当出现显式随意运动时才可提取；必须为具体动作；模糊描述若无具体动作支撑，不得提取。",
                        "  • freeze_or_faint：仅当出现显式冻结或晕厥倾向行为时才可提取；主观体验不得作为依据；必须为可观测的身体状态变化。",
                        "  • intensity：仅当原文出现强度修饰词时才可量化；映射规则：'剧烈'/'完全'→1.00，'明显'/'大幅'→0.70，'微微'/'轻轻'→0.30；无此类词则省略该字段。",
                        "  • evidence：必须为包含整个躯体行为及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要或 paraphrasing；可忽略前导及尾随空白差异。",
                        "  • semantic_notation：每个包含任一有效躯体化要素的事件必须包含此字段；格式为 {bodily_category}_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - bodily_category 必须为以下之一：facial, vocal, postural, locomotor, manual, autonomic_visible, freeze；",
                        "    - english_summary 必须是一句高度提炼的英文躯体行为概括，准确表达该可观测行为的核心语义；",
                        "    - 该概括所依赖的所有关键要素（动作、部位、方向、强度、可见征象等）必须在当前事件的 evidence 中有显式文字支持；禁止虚构、推理、补充常识或引入未出现的概念；",
                        "    - 禁止包含情绪词、医学术语、内部感受、代词（如 'it', 'that'）、品牌、坐标值、中文、拼音或系统提示占位词；",
                        "    - 若无法生成合规摘要，则使用 bodily_mentioned。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 bodily 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "bodily": {
                        "type": "object",
                        "items": {
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
                "role": "你是一个严格遵循结构契约的情感状态信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的情感状态描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "emotional",
                "constraint_profile": "high_fidelity_emotional_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【情绪提取核心原则】——————
                        "- 【情绪词字面强制】emotion_labels 必须为原文中逐字出现的情绪词或情绪修饰短语（如‘焦虑’‘悲痛欲绝地说’‘带着怒意’‘好烦啊’）；不得替换、拆分、同义转换、抽象化、标准化或映射至任何情绪模型。",
                        "- 【行为≠情绪】哭泣、跺脚、沉默、叹气、脸红、转身、颤抖、握拳等行为、生理反应或躯体表现不得作为情绪依据；仅当字面出现情绪形容词、情绪副词、情绪名词或含情绪语义的固定短语时才可提取 emotion_labels。",
                        "- 【混合语言保真】若原文混用中英文情绪表达（如‘他很 jiaolv’‘她 feel sad’），emotion_labels 保留原形式；不得强行翻译、转写或标准化。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式、引号、语气助词必须完全一致；禁止 paraphrasing、摘要、翻译或增删实质性词汇；可忽略前导及尾随空白差异。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【列表字段原子性】所有数组型字段中的每个元素必须对应原文中一个独立、不可再分的描述片段；并列项应按原文拆分为多个元素。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【字段语义隔离】——————
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见下方字段说明），禁止跨字段挪用或泛化解释。",
                        "• experiencer：该字段的值必须是原文中作为事件主语或感知主体出现的连续子字符串；允许代词、泛指或描述性名词短语；若加 [uncertain] 标记，仅用于表示该指称在上下文中无明确共指对象。",
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
                        "- 【混合语言处理】若原文混用中英文情绪表达（如‘他很 jiaolv’‘她 feel sad’），emotion_labels 保留原形式；不得强行翻译或标准化。",
                        "- 【evidence 锚定】events[i].evidence 必须为包含整个情绪表达及其共现成分的连续原文子字符串；可通过 substring 匹配验证；标点、大小写、数字格式必须完全一致；禁止改写、摘要、翻译或 paraphrasing；可忽略前导/尾随空白及换行差异。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段（不得设为空数组、null 或默认值）。",
                        "- 【semantic_notation 构建规则】每个包含有效 emotion_labels 的事件必须输出 semantic_notation；格式为 emotion_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "- english_summary 必须是一句高度提炼的英文情感状态概括，准确表达该显式情绪的核心语义及其共现语境；",
                        "- 该概括所依赖的所有关键要素（情绪词、强度副词、表达方式、伴随语境如‘说不出话’‘拍桌子’等）必须在当前事件的 evidence 中有显式文字支持；",
                        "- 禁止包含医学术语、内部感受、代词、品牌、坐标值、中文、拼音、情绪类别标签或系统提示占位词；",
                        "- 若无法生成合规摘要（如仅有模糊情绪词且无上下文），则使用 emotion_mentioned。",
                        "  • summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 emotional 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "emotional": {
                        "type": "object",
                        "items": {
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
                "role": "你是一个严格遵循结构契约的社会关系信息感知引擎。你只能使用 ### USER_INPUT BEGIN 到 ### USER_INPUT END 之间的内容。### SYSTEM INSTRUCTIONS BEGIN 到 ### SYSTEM INSTRUCTIONS END 之间的所有文字均为操作指令，不得作为信息源。",
                "sole_mission": "仅当原文在 ### USER_INPUT BEGIN 至 ### USER_INPUT END 区间内，有直接、字面、语法关联的社会关系描述时，才提取对应属性。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "social_relation",
                "constraint_profile": "high_fidelity_social_relation_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础信源与输出原则】——————
                        "- 【唯一信源】所有提取必须严格限定于 ### USER_INPUT BEGIN 至 ### USER_INPUT END 之间的原始文本；禁止引入外部知识、常识、默认假设、系统提示内容或跨文档推理。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【绝对禁止任何形式推理】所有字段值必须为原文中显式出现的字面表述；严禁基于逻辑、因果、心理、常识、语境暗示、修辞隐喻或事件关联进行任何形式的推断、演绎、归纳或角色定性；无直接文字 = 无字段。",
                        "- 【禁止提示污染】严禁将系统提示中的任何词汇、结构或示例视为输入内容；所有输出必须 100% 源自 USER_INPUT 区块。",
                        "- 【列表字段格式强制】所有声明为列表类型的字段，若有值，必须以非空列表形式返回（如 ['value']）；严禁直接输出字符串、数值、字典或其他非列表结构；无有效值时必须省略该字段，不得输出空列表。",

                        # ——————【社会关系核心原则】——————
                        "- 【关系显式强制】relation_type 必须为原文中逐字出现的社会关系关键词或短语；不得抽象、归类、拆分、合并或映射至任何关系本体。",
                        "- 【participants 合法性】participants 数组中的每个元素必须是原文中在关系陈述里显式出现的指称短语；若该指称为代词或泛指，且上下文中无明确对应的完整 noun phrase，则必须表示为 '<原文指称>[uncertain]'；若为具体 noun phrase 或专有名称，则直接复制。",
                        "- 【participants 最小数量】每个事件的 participants 数组必须包含至少两个元素（可含 [uncertain] 标记）；仅当这两个指称均在同一关系陈述中作为语法成分显式出现时，方可提取。",
                        "- 【explicit_relation_statement 要求】每个事件必须对应原文中一个语法自足、语义完整的显式关系陈述；不得通过拼接跨句信息、截取半句话或重构主谓宾来构造关系；关系陈述必须在单一句子或连续短语内完成闭环。",

                        # ——————【evidence 与字段存在性】——————
                        "- 【evidence 锚定】evidence 必须为原文中的连续子字符串，可通过 substring 匹配验证；标点、大小写、数字格式、引号必须完全一致；禁止 paraphrasing、摘要、翻译、增删词汇或重构句式；可忽略前导及尾随空白差异。",
                        "- 【字段存在性】除 evidence 外，所有字段仅当原文中有直接、字面、语法共现的显式依据时才可出现；无锚定则彻底省略该字段。",

                        # ——————【experiencer 提取与标记规则】——————
                        "- 【experiencer 提取优先级】按以下顺序确定 experiencer：",
                        "  1. 优先提取与事件主语显式共指的具体 noun phrase（如前句主语、同位语、重复出现的完整描述）；",
                        "  2. 若无具体 noun phrase，则提取代词作为 experiencer，并标记为 '<代词>[uncertain]'；",
                        "  3. 仅在完全无主体的事件中才省略 experiencer 字段；",
                        "- 【具体指称处理】若 experiencer 为具体 noun phrase 或专有名称，则直接复制，不加任何标记；",
                        "- 【代词标记统一规则】以下情况必须添加 [uncertain] 标记：",
                        "  • 事件主语为代词（如'他''我'），且无法找到无歧义对应的具体 noun phrase；",
                        "  • 事件主语为泛指（如'有人''一个人'）；",
                        "  • 从上下文中无法确定具体指称的代词；",
                        "- 【标记完整性】禁止仅输出裸代词而不标记不确定性；所有代词 experiencer 必须包含 [uncertain] 标记；",
                        "- 【无主体判定】仅当事件无显式主语且无法从语法结构推断感知主体（如无人称句、纯客观陈述）时，才可省略 experiencer 字段；",

                        # ——————【关系提取纪律】——————
                        "- 【relation_type 单一性】每个事件仅描述一种显式关系；若原文一句包含多个独立关系（如‘他是我哥哥，也是李姐的前男友’），应拆分为多个事件；若为固定复合词（如‘发小’‘死党’），则整体作为一个 relation_type 保留。",
                        "- 【禁止间接推断】不得从称呼语、动作行为、空间共现、对话内容、情感表达、职业标签等非显式陈述中推导社会关系；仅当原文出现完整关系谓词且包含明确关系词时，才可提取。",

                        # ——————【字段语义与提取纪律】——————
                        "- 【禁止常识推理与属性脑补】所有字段值必须有原文的直接、字面陈述作为唯一依据；严禁基于衣着、行为、名字、职业、文化默认、性别刻板印象、逻辑推断或概率猜测进行任何属性填充。若原文未明确说出某属性，则该字段必须彻底省略。",
                        "- 【字段语义严格隔离】各属性字段仅在其明确定义条件下提取（详见字段规范），禁止跨字段挪用、泛化解释或引入未出现的概念。",
                        "- 【零合成、零概括、零标准化】禁止合并、替换、修正原文表述；所有值必须为原文逐字片段，不得增删改。",

                        # ——————【semantic_notation 构建规则】——————
                        "- 【semantic_notation 强制输出】每个包含有效 participants（≥2）和 relation_type 的事件必须输出 semantic_notation；格式为 relation_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "  - english_summary 必须是一句高度提炼的英文社会关系概括，准确表达该显式关系的核心语义及其共现语境；",
                        "  - 该概括所依赖的所有关键要素（关系词、修饰语、参与者身份等）必须在当前事件的 evidence 中有显式文字支持；",
                        "  - 禁止包含动词、否定、泛化标签、心理推断、虚构术语、人名、品牌、坐标值、中文、拼音、情绪词或系统提示占位词；",
                        "  - 若无法生成合规摘要（如仅有模糊关系词且无上下文），则使用 relation_mentioned。",

                        "- summary：不超过100字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 social_relation 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 spatial 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "social_relation": {
                        "type": "object",
                        "items": {
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
                "role": (
                    "你是一个严格受限的因果与动机推理引擎，仅能访问以下三个输入区块："
                    "(1) ### PARTICIPANTS_VALID_INFORMATION BEGIN 与 ### PARTICIPANTS_VALID_INFORMATION END之间（参与者有效数据）；"
                    "(2) ### PERCEPTUAL_CONTEXT_BATCH BEGIN 与 ### PERCEPTUAL_CONTEXT_BATCH END之间（已验证的感知事件有效数据）；"
                    "(3) ### LEGITIMATE_PARTICIPANTS BEGIN 与 ### LEGITIMATE_PARTICIPANTS END之间（合法的参与者实体，用于 experiencer 字面匹配）；"
                ),
                "sole_mission": "基于已验证的感知事件、参与者有效信息和合法的参与者实体，生成最小必要、最大保真、可追溯、弱化表述的推演命题。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "inference",
                "constraint_profile": "high_fidelity_inference_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        "- 【输入范围】仅可使用以下三块数据：(1) PARTICIPANTS_VALID_INFORMATION；(2) PERCEPTUAL_CONTEXT_BATCH；(3) LEGITIMATE_PARTICIPANTS；不得引用系统提示、外部知识或用户原始文本。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出'未提及'、'未知'、空字符串、null 或空列表。",
                        "- 【禁止提示污染】严禁将系统提示中的任何内容视为有效输入。",
                        "- 【字段存在性】除 anchor_points 和 inferred_proposition 外，所有字段仅当有直接、字面、可验证的锚定时才可出现；无锚定 = 彻底省略。",
                        "- 【推演来源封闭性】所有推理必须 100% 源自 PERCEPTUAL_CONTEXT_BATCH 中已输出的感知事件；严禁复用系统示例、模板填充或外部知识；若感知层无输出，则推演层返回 {}。",
                        "- 【多感知协同推演允许】推演命题可基于 PERCEPTUAL_CONTEXT_BATCH 中任意数量的感知事件联合构建，允许多模态、跨类型融合分析。",
                        "- 【experiencer 合法性强化】experiencer 必须同时满足：",
                        "  • 字面匹配：严格等于 LEGITIMATE_PARTICIPANTS 中的完整 noun phrase或代词指称；",
                        "  • 证据对应：必须与所引用感知事件的原始 experiencer 保持语义一致性；",
                        "  • 逻辑自洽：必须与 inferred_proposition 的心理主体和 anchor_points 的参与者维度一致；",
                        "  • 禁止混淆：严禁跨参与者混合推理或将观察者误推为体验者；",
                        "- 【inference_type 枚举强制】inference_type 必须为预定义枚举值之一：causal, temporal, intentional, belief, contradiction, consistency_check, counterfactual, abductive, normative, social_attribution, state_transition, predictive, analogical；不得自创、缩写或替换。",
                        "- 【anchor_points 与 evidence 的同源绑定】推演事件中的 anchor_points 和 evidence 必须构成一一对应的引用对列表，其中每一对 (anchor_points[j], evidence[j]) 必须严格源自 PERCEPTUAL_CONTEXT_BATCH 中同一个感知事件的 (semantic_notation, evidence)；禁止跨事件错配、字段孤立引用或虚构绑定。",
                        "- 【anchor_points 来源限制】anchor_points 中的每个元素必须完全等于 PERCEPTUAL_CONTEXT_BATCH 中某个感知事件的 semantic_notation；不得泛化、合成或引用未输出的标签；若为空，则推理事件无效。",
                        "- 【evidence 锚定】evidence 中的每个元素必须完全等于 PERCEPTUAL_CONTEXT_BATCH 中对应感知事件的原始 evidence 字符串；标点、大小写、格式必须一致；可通过 substring 验证；禁止改写、摘要或 paraphrasing。",
                        "- 【inferred_proposition 弱化强制】inferred_proposition 必须以非确定性引导词开头（如'可能''似乎''或许'），全文不得包含强断言词（如'一定''证明''说明'）；内容须为单一、可证伪命题，长度 ≤100 字；不得引入未在所引用感知事件中隐含的新实体、关系或属性。",
                        "- 【semantic_notation 构建规则】仅当 anchor_points 非空且 inferred_proposition 存在时，必须输出 semantic_notation；格式为 inference_{english_summary}（≤128字符，snake_case）；",
                        "    - english_summary 必须是一句高度提炼的英文推理核心事件概括，准确表达该推演命题核心语义及其共现语境（如因果、意图、状态变迁等）；",
                        "    - 其所有关键要素必须能在所引用的感知事件的 evidence 或 semantic_notation 中找到直接或结构性支持；",
                        "    - 禁止包含心理诊断术语、抽象哲学概念、虚构动词结构、人名、品牌、坐标、情绪标签、泛化代词或系统占位词；",
                        "    - 若无法生成合规摘要，则使用 inference_mentioned。",
                        "- 【polarity 枚举与锚定强制】polarity 必须且仅可为以下三个预定义值之一：positive, negative, neutral；该字段仅当 inferred_proposition 或其引用的 evidence 中显式包含可客观验证的正面/负面/无评价倾向时才可出现；否则必须彻底省略。",
                        "- 【context_modality 显式触发】context_modality（factual/hypothetical/obligatory/permitted/prohibited）仅当所引用的 evidence 中包含显式情态词（如'应该''如果''必须''禁止'）时才可出现；其值必须与原文情态语义严格对应；否则省略。",
                        "- 【scope 客观界定】scope（individual/group/institutional/cultural）仅当 inferred_proposition 或 experiencer 明确指向某类实体范围时才可出现；不得从个体行为推测群体或文化属性；否则省略。",
                        "- summary：不超过200字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 inference 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 inference 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤200字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。",
                        "- 【inference_type 枚举强制】必须为预定义13种类型之一，不可扩展。",
                        "- 【inferred_proposition 弱化强制】必须以‘可能’‘似乎’等非确定性词开头，≤100字，禁用强断言词。",
                        "- 【context_modality 触发条件】仅当原文含显式情态词时才可出现；否则省略。",
                        "- 【极性与范围】polarity 和 scope 若存在，须可从 inferred_proposition 或 experiencer 客观导出，不得主观推断。"
                    ]
                },
                "fields": {
                    "inference": {
                        "type": "object",
                        "items": {
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
                "role": (
                    "你是一个严格受限的显性动机提取引擎，仅能访问以下四个输入区块："
                    "(1) ### PARTICIPANTS_VALID_INFORMATION_BEGIN 与 ### PARTICIPANTS_VALID_INFORMATION_END之间（参与者有效数据）；"
                    "(2) ### PERCEPTUAL_CONTEXT_BATCH_BEGIN 与 ### PERCEPTUAL_CONTEXT_BATCH_END之间（已验证的感知事件有效数据）；"
                    "(3) ### LEGITIMATE_PARTICIPANTS_BEGIN 与 ### LEGITIMATE_PARTICIPANTS_END之间（合法的参与者实体，用于 experiencer 字面匹配）；"
                    "(4) ### INFERENCE_CONTEXT_BEGIN 与 ### INFERENCE_CONTEXT_END之间（已验证的合理推演层数据）；"
                ),
                "sole_mission": "基于已验证的感知事件、参与者有效信息、合法的参与者实体和已验证的合理推演层数据，生成最小必要、最大保真、可追溯、弱化表述的显性动机命题。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "explicit_motivation",
                "constraint_profile": "high_fidelity_explicit_motivation_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        "- 【输入范围】仅可使用以下四块数据：(1) PARTICIPANTS_VALID_INFORMATION；(2) PERCEPTUAL_CONTEXT_BATCH；(3) LEGITIMATE_PARTICIPANTS；(4) INFERENCE_CONTEXT。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出‘未提及’、‘未知’、空字符串、null 或空列表。",
                        "- 【禁止提示污染】严禁将系统提示中的任何内容视为有效输入。",
                        "- 【experiencer 合法性强化】experiencer 必须同时满足：",
                        "  • 字面匹配：严格等于 LEGITIMATE_PARTICIPANTS 中的完整 noun phrase或代词指称；",
                        "  • 证据对应：必须与所引用感知事件的原始 experiencer 保持语义一致性；",
                        "  • 逻辑自洽：必须与 inferred_proposition 的心理主体和 anchor_points 的参与者维度一致；",
                        "  • 禁止混淆：严禁跨参与者混合推理或将观察者误推为体验者；",
                        "- 【动机显性要求】仅可提取叙述者对角色内心状态或行为目的的直接断言，包括：第一人称心理陈述；第三人称意图归因；具象化描写中明确绑定意图的心理归因。",
                        "- 【禁止提取】纯行为描述；隐喻/象征/氛围渲染；反问、讽刺、暗示、留白、推测性语言；系统示例或通用模板均不得作为动机依据。",
                        "- 【evidence 锚定】events[i].evidence 必须为用户原始输入中的最小连续原文片段，且该片段必须包含对心理状态或行为目的的明确断言；标点、大小写、数字格式必须完全一致；可通过 substring 验证；禁止改写、概括、翻译、增删实质性词汇；允许忽略前导/尾随空白及换行差异。",
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
                        "- 【motivation_category 枚举与锚定】motivation_category 必须为预定义枚举值之一（fear, care, power, survival, social_norm, self_deprecation）；且必须有 evidence 中的直接语义支撑；禁止仅因 core_driver 存在就自动设为 fear；若无明确分类依据，则省略该字段。",
                        "- 【semantic_notation 构建规则】仅当事件非空且能生成合规英文摘要时，才输出 semantic_notation；格式为 explicit_motivation_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - english_summary 必须是一句高度提炼的英文显性动机事件概括，准确表达该显性动机核心意图及其共现语境（如避免抛弃、保护子女、通过资源控制维持服从等）；",
                        "    - 其所有关键要素（动词、目标、条件、关系）必须能在所引用的 evidence 片段中找到直接或结构性支持；",
                        "    - 禁止包含心理诊断术语、抽象哲学概念、虚构动词结构、人名、品牌、坐标、情绪标签、泛化代词或系统占位词；",
                        "    - 若无法生成合规摘要（如动机表述模糊、缺乏具体目标），则不输出 semantic_notation 字段。",
                        "- summary：不超过200字，用通顺的中文自然语言客观陈述核心事件，不得添加评价、推测或无关细节。"
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，仅包含 explicit_motivation 字段，无任何额外文本、注释、markdown 或格式装饰。",
                        "- 【顶层省略规则】若 events 为空，则 temporal 对象整体省略（返回 {}）。",
                        "- 【顶层字段强制】若 events 非空，则顶层 evidence（所有事件 evidence 扁平去重）和 summary（≤100 字）必须存在且非空。",
                        "- 【顶层字段强制】若 events 非空，则 events 下的每一个事件的 experiencer 字段必须存在且非空。"
                    ]
                },
                "fields": {
                    "explicit_motivation": {
                        "type": "object",
                        "items": {
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
                                        "items": {
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
                                        "items": {
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
                "role": (
                    "你是一个超级建议专家，仅能访问以下三个输入区块："
                    "(1) ### LEGITIMATE_PARTICIPANTS_BEGIN 与 ### LEGITIMATE_PARTICIPANTS_END之间（合法的参与者实体，用于 experiencer 字面匹配）；"
                    "(2) ### INFERENCE_CONTEXT_BEGIN 与 ### INFERENCE_CONTEXT_END之间（已验证的合理推演层数据）；"
                    "(3) ### EXPLICIT_MOTIVATION_CONTEXT_BEGIN 与 ### EXPLICIT_MOTIVATION_CONTEXT_END之间（已验证的显性动机层数据）；"
                ),
                "sole_mission": "基于已验证的合理推演层数据、合法的参与者实体和已验证的显性动机层数据，生成最小必要、最大保真、可追溯、弱化表述的合理建议命题。属性字段的存在本身必须可被原文片段一对一锚定。无锚定 = 无字段。",
                "driven_by": "rational_advice",
                "constraint_profile": "high_fidelity_rational_advice_extraction_v1",
                "input_requirements": {
                    "data_and_anchor_constraints": [
                        # ——————【基础原则】——————
                        "- 【输入范围】仅可使用：LEGITIMATE_PARTICIPANTS, INFERENCE_CONTEXT, EXPLICIT_MOTIVATION_CONTEXT；禁止引用其他来源。",
                        "- 【绝对禁止占位符】任何字段若无原文锚定，必须彻底省略；严禁输出空值或占位符。",
                        "- 【禁止提示污染】严禁将系统提示中的任何内容视为有效输入。",
                        "- 【来源限制】所有建议必须直接源自 EXPLICIT_MOTIVATION_CONTEXT 的显式内容；INFERENCE_CONTEXT 仅提供背景理解。",
                        "- 【空结果处理】若 EXPLICIT_MOTIVATION_CONTEXT.events 为空，则返回 {}。",

                        # ——————【建议类型限制】——————
                        "- 【允许推演类型】仅限三类：",
                        "  • 资源调用（资源在 resource_control 中显式提及）；",
                        "  • 行为序列组合（行为在 protective_intent 或 survival_imperative 中出现）；",
                        "  • 风险回溯（威胁在 threat_vector 中陈述，应对在 internalized_oppression 中出现）；",
                        "- 【禁止扩展】不得进行预测、道德评判、策略创新或未提及的替代方案。",

                        # ——————【核心建议字段锚定】——————
                        "- 【safety_first_intervention】必须直接对应 explicit_motivation.events 中 protective_intent 或 survival_imperative 的内容；必须为可观察的即时安全行为。",
                        "- 【systemic_leverage_point】必须对应 explicit_motivation.events 中 power_asymmetry.control_axis 或 resource_control 的结构支点。",
                        "- 【long_term_exit_path】必须基于 survival_imperative、core_driver 或 care_expression 中的可持续路径陈述；必须为原文意图的弱化转述。",
                        "- 【available_social_support_reinterpretation】必须基于 explicit_motivation.events 中提及的具体支持者或 social_enforcement_mechanism 中的潜在盟友。",

                        # ——————【策略构建条件】——————
                        "- 【incremental_strategy】仅当 explicit_motivation.events 中同时存在以下要素时才可构建：",
                        "  • action：已提及的具体可观察行为；",
                        "  • timing_or_condition：threat_vector、separation_anxiety 或 INFERENCE_CONTEXT 的触发情境；",
                        "  • required_resource：字面出现的资源；",
                        "  • potential_risk：threat_vector 或 narrative_distortion 的压迫者负面反应；",
                        "  • contingency_response：internalized_oppression 或 self_justification 的历史退让行为；",

                        "- 【fallback_plan】仅当同时存在以下要素时才可构建：",
                        "  • trigger_condition：threat_vector、separation_anxiety 或 narrative_distortion.blame_shift 的风险信号；",
                        "  • fallback_action：internalized_oppression、self_justification 或 survival_imperative 的历史退让行为；",

                        # ——————【利益相关者分析】——————
                        "- 【stakeholder_tradeoffs】仅当任一子字段有内容时才可出现：",
                        "  • victim_cost：必须引用 core_driver 或 survival_imperative 的 evidence 片段；",
                        "  • oppressor_loss：必须对应 power_asymmetry.control_axis 或 resource_control 的 semantic_notation；",
                        "  • system_stability：仅当存在 social_enforcement_mechanism 时才可填充；",
                        "  • evidence：必须列出引用的 semantic_notation；",

                        # ——————【可执行性要求】——————
                        "- 【动作可观测性】每条建议必须包含：",
                        "  • 具体行为主体（LEGITIMATE_PARTICIPANTS 中的实体）；",
                        "  • 可观测动作（动词+宾语）；",
                        "  • 明确的操作条件或触发情境；",
                        "- 【禁止抽象指令】如'增强自信''寻求帮助'等不得作为建议。",

                        "- 【semantic_notation 构建规则】仅当 rational_advice 非空且能生成合规英文摘要时，才输出 semantic_notation；格式为 rational_advice_{english_summary}（总长度不超过128字符，全小写snake_case）；",
                        "    - english_summary 必须是对整体建议策略的一句英文高度概括，准确表达该建议的合理、可执行性和必要性等核心内容；",
                        "    - 其所有关键要素（目标、手段、条件、资源、风险）必须能在 EXPLICIT_MOTIVATION_CONTEXT 的字段内容或 evidence 中找到直接或结构性支持；",
                        "    - 禁止包含心理术语、道德判断、抽象概念、人名、品牌、坐标、情绪标签、泛化动词（如 do_something）或系统占位词；",
                        "    - 若无法生成合规摘要（如建议过于碎片化或缺乏统一逻辑），则不输出 semantic_notation 字段。",
                        "- 【summary 规则】≤200 字，仅中文客观复述最小可行路径，使用第三人称、被动语态或行为清单式表述（如‘可考虑联系母亲；保存聊天记录；在对方饮酒后避免冲突’）。",
                    ],
                    "output_structure_constraints": [
                        "- 【JSON 纯净性】仅返回合法 JSON，无任何额外文本、注释、Markdown 或格式装饰。",
                        "- 【顶层省略规则】若无有效建议，则 rational_advice 整体省略（返回 {}）。",
                        "- 【顶层字段强制】若建议非空，则 evidence 和 summary 必须存在且非空。",
                        "- 【建议精简】建议数量控制在 2-5 条，避免冗余。"
                    ]
                },
                "fields": {
                    "rational_advice": {
                        "type": "object",
                        "items": {
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
                                "items": {
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
    COREFERENCE_RESOLUTION_BATCH: (
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
