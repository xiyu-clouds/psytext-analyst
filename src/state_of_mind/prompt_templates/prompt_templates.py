"""存放所有阶段模板定义"""

from src.state_of_mind.stages.perception.constants import CATEGORY_RAW, LLM_PARTICIPANTS_EXTRACTION, \
    LLM_PERCEPTION_TEMPORAL_EXTRACTION, LLM_PERCEPTION_SPATIAL_EXTRACTION, \
    LLM_PERCEPTION_VISUAL_EXTRACTION, LLM_PERCEPTION_GUSTATORY_EXTRACTION, LLM_PERCEPTION_TACTILE_EXTRACTION, \
    LLM_PERCEPTION_OLFACTORY_EXTRACTION, LLM_PERCEPTION_AUDITORY_EXTRACTION, LLM_PERCEPTION_EMOTIONAL_EXTRACTION, \
    LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION, LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION, \
    LLM_PERCEPTION_COGNITIVE_EXTRACTION, LLM_PERCEPTION_BODILY_EXTRACTION, \
    CATEGORY_SUGGESTION, COREFERENCE_RESOLUTION_BATCH, \
    LLM_DIMENSION_GATE, LLM_INFERENCE_ELIGIBILITY, GLOBAL_SEMANTIC_SIGNATURE, LLM_STRATEGY_ANCHOR, \
    LLM_CONTRADICTION_MAP, LLM_MANIPULATION_DECODE, LLM_MINIMAL_VIABLE_ADVICE, PARALLEL_PREPROCESSING, \
    PARALLEL_PERCEPTION, PARALLEL_HIGH_ORDER, SERIAL_SUGGESTION

LLM_PROMPTS_SCHEMA = {
    CATEGORY_RAW: {
        # === 控制流：处理阶段划分 ===
        "pipeline": [
            # 提取参与者
            {
                "step_name": LLM_PARTICIPANTS_EXTRACTION,
                "type": PARALLEL_PREPROCESSING,
                "index": 0,
                "label": "预处理：大模型参与者信息提取",
                "role": "你是一个高保真结构化提取器。你的唯一任务是根据用户输入的字面文本，提取并输出结构化数据。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止直接使用本指令中的示例、模板、说明文字或任何外部知识。",
                "driven_by": "participants",
                "constraint_profile": "high_fidelity_participant_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**。"
                ],
                "output_suffix": [],
                "empty_result_fallback": '若未识别到任何实体，则输出：`{"participants": []}`。',
                "step_rules": [
                    "- **属性提取边界**：",
                    "  • entity：仅当原文中出现以下任一形式时，方可提取一个参与者实体：1) 指代人物的专有名词（如姓名）、社会身份标签（如“医生”）或带具体修饰语的描述性名词短语（如“穿红衣服的男人”）；2) 若全文未出现上述任何非代词指称，且仅通过人称代词（如“他”、“她”）指代某人物，则可提取该代词作为 entity。该实体的所有属性值，必须源自原文中**直接修饰该 entity 指称或其对应代词**的描述性成分——即：描述性词语或短语在句子结构上必须以该 entity（或其代词）作为主语、宾语、定语或所有格中心词。不满足此条件的描述，一律禁止关联至该实体。",
                    "  • social_role：仅当原文使用非职业的社会身份标签（如‘父亲’、‘朋友’）并通过直接修饰或系动词关联 entity 时才提取。",
                    "  • occupation：仅当原文使用职业身份词（如‘教师’、‘工程师’）并直接修饰或通过系动词（如‘是’、‘担任’）连接 entity 指称时才提取。",
                    "  • family_status：仅当原文使用家庭状态标签（如‘已婚’、‘单身’）并直接陈述 entity 状态时才提取。",
                    "  • education_level：仅当原文出现明确教育程度表述（如‘博士’、‘高中毕业’）并关联 entity 时才提取。",
                    "  • cultural_identity：仅当原文明确使用文化/民族身份标签（如‘汉族’、‘穆斯林’）并指称 entity 时才提取。",
                    "  • primary_language：仅当原文陈述 entity 使用的主要语言（如‘说英语’、‘母语是中文’）时才提取。",
                    "  • institutional_affiliation：仅当原文提及 entity 所属组织（如‘就职于XX公司’、‘是XX大学学生’）时才提取。",
                    "  • age_range：仅当原文出现‘青年/少年/中年/老年’等词并直接用于 entity 时才可提取。",
                    "  • gender：仅当原文出现‘男/女/男性/女性’等词并直接修饰或指称 entity 时才可提取；代词‘他/她’不构成依据。",
                    "  • ethnicity_or_origin：仅当原文出现‘来自[地点]’、‘[地域]籍’、‘生于[地]’、‘[民族]族’等结构并直接关联 entity 指称时才提取。",
                    "  • physical_traits：仅当原文描述 entity 的身体部位、体型、肤色等非临时性状态的形容词或名词短语时才提取。",
                    "  • current_action_state：仅限原文中描述 entity 当前身体姿势、动作或直接生理反应的动词短语或形容词（如‘坐着’、‘奔跑’、‘呼吸急促’）。",
                    "  • visible_injury_or_wound：仅当原文明确描述 entity 身体上的可见伤痕、疤痕或医疗痕迹时才提取。",
                    "  • appearance：仅限原文中描述 entity 的服装、发型、配饰、面部特征等外观的形容词或名词短语。",
                    "  • voice_quality：仅当原文直接描述 entity 的嗓音特征（如‘沙哑’、‘清脆’）时才提取。",
                    "  • inherent_odor：仅当原文直接描述 entity 具有某种气味时才提取。",
                    "  • possessions：仅当原文明确提及 entity 物理持有或佩戴的具体物品时才提取。",
                    "  • speech_pattern：仅当原文明确提及口音、语速、用词习惯等（如‘说话带南方口音’）时才提取。",
                    "  • interaction_role：仅当原文使用‘凶手’、‘目击者’、‘受害者’等事件角色词并明确指派给 entity 时才提取。"
                ],
                "fields": {
                    "participants": {
                        "type": "array",
                        "items": {
                            "entity": { "type": "string" },
                            "social_role": { "type": "string" },
                            "occupation": { "type": "string" },
                            "family_status": { "type": "string" },
                            "education_level": { "type": "string" },
                            "cultural_identity": { "type": "string" },
                            "primary_language": { "type": "string" },
                            "institutional_affiliation": { "type": "string" },
                            "age_range": { "type": "string" },
                            "gender": { "type": "string" },
                            "ethnicity_or_origin": { "type": "string" },
                            "physical_traits": { "type": "array" },
                            "current_action_state": { "type": "string" },
                            "visible_injury_or_wound": { "type": "string" },
                            "appearance": { "type": "array" },
                            "voice_quality": { "type": "string" },
                            "inherent_odor": { "type": "string" },
                            "possessions": { "type": "array" },
                            "speech_pattern": { "type": "string" },
                            "interaction_role": { "type": "string" }
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 动态决定启用哪些感知步骤
            {
                "step_name": LLM_DIMENSION_GATE,
                "type": PARALLEL_PREPROCESSING,
                "index": 1,
                "label": "预处理：动态筛选值得调用的感知维度",
                "role": "你是一个感知线索分类器。你的任务是根据用户输入，判断其中是否明确包含属于以下12个感知维度的字面描述。",
                "information_source": "**判断依据**：每个维度的 `true`/`false` 判断必须严格基于 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本中的字面描述。禁止直接使用本指令中的示例、模板、说明文字或任何外部知识。",
                "driven_by": "pre_screening",
                "constraint_profile": "high_fidelity_perceptual_gate_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**（注意保留顶层键 \"pre_screening\"）："
                ],
                "output_suffix": [
                    "**核心判断规则**：",
                    "- **字面依据**：判断必须基于输入文本中直接存在的词汇或短语。使用其最直接、常见的字面含义。",
                    "- **比喻排除原则**：对于明确使用“像”、“如”、“仿佛”等比喻词引导的描述，其喻体通常不单独作为触发感知维度的依据。",
                    "- **存在即真**：只要找到一个明确符合上述定义的描述，该维度即为 `true`。",
                    "- **独立与重叠判断**：每个维度的判断独立进行。**如果一个描述同时符合多个维度的定义，则所有相关维度均应标记为 `true`。**",
                    "- **无即假**：如果输入中没有找到任何符合定义的描述，则该维度为 `false`。"
                ],
                "empty_result_fallback": "",
                "step_rules": [
                    "**感知维度定义与关键示例**：",
                    "1.  **temporal (时间)**：明确提及时间点、时段、日期或时序关系。【例】“夜晚九点”、“十分钟后”、“去年”。【非例】“时机成熟”（隐喻）。",
                    "2.  **spatial (空间)**：明确提及地点、方位、距离、布局或空间关系。【例】“客厅里”、“阳台上”、“远处”。【非例】“心田”（隐喻）。",
                    "3.  **visual (视觉)**：明确描述肉眼可见的外观、景象、动作、颜色、物体状态或场景。【例】“穿着白色短袖”、“月光洒下”、“睫毛低垂”。",
                    "4.  **auditory (听觉)**：明确描述声音、声响、语音特征或静默。【例】“声音发颤”、“沉默”、“哽咽”。",
                    "5.  **olfactory (嗅觉)**：明确描述气味或闻的动作。【例】“闻到花香”、“气味”。",
                    "6.  **tactile (触觉)**：明确描述物理接触感、温度、质地或压力。【例】“微风轻拂”、“摩挲”、“冰冷”。【注意】区分字面描述与比喻。",
                    "7.  **gustatory (味觉)**：明确描述味道或品尝动作。【例】“尝起来甜”、“苦涩”。",
                    "8.  **interoceptive (内感受)**：明确描述身体内部生理感觉。【例】“心跳加速”、“肚子饿”、“胸口发闷”、“哽咽”。【非例】“心碎”（比喻）。",
                    "9.  **cognitive (认知)**：明确描述思考、记忆、知道、怀疑、相信等心智活动。【例】“认为”、“记得”、“想知道”。",
                    "10. **bodily (身体动作)**：明确描述具体的、中性的身体部位运动或姿态。【例】“坐着”、“站着”、“挽起头发”、“摩挲”。（与visual独立判断）",
                    "11. **emotional (情绪)**：明确使用**公认的情绪词汇或短语**直接陈述主体的情绪状态。【例】“无奈”、“愤怒”、“高兴”。【非例】“她哭了”（行为，除非明确“悲伤地哭了”）。",
                    "12. **social_relation (社会关系)**：明确提及人与人之间的特定关系、角色或互动称谓。【例】“母亲”、“女儿”、“同事”、“陪”。",
                ],
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
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 提前判断文本是否值得启用高阶推理
            {
                "step_name": LLM_INFERENCE_ELIGIBILITY,
                "type": PARALLEL_PREPROCESSING,
                "index": 2,
                "label": "预处理：判断是否触发高阶四步（策略/矛盾/操控/建议）",
                "role": "你是一个高阶语义守门员。你的唯一任务是判断### USER_INPUT BEGIN###与### USER_INPUT END###之间的文本是否包含策略性、矛盾性、操控性或可干预性要素，从而决定是否值得启动完整的高阶四步分析链。",
                "information_source": "",
                "driven_by": "eligibility",
                "constraint_profile": "inference_eligibility_abstract_strict_v1",
                "output_prefix": [
                    "你必须输出一个符合以下 JSON Schema 的**数据实例**，而不是输出 Schema 本身（注意保留顶层键 \"eligibility\"）："
                ],
                "output_suffix": [],
                "empty_result_fallback": '',
                "step_rules": [
                    "### 核心判断规则",
                    "仅当用户输入中**直接包含**或通过**日常对话惯例明确暗示**以下任一高阶特征时，才判定 eligible = true：",

                    "1. **策略性行为**：",
                    "   - 明确表达**目标 + 手段**（如“我想让他离职，所以先收集他的把柄”）",
                    "   - 描述**分阶段计划**、**资源利用**、**风险规避**或**长期布局**。",

                    "2. **内在或人际矛盾**：",
                    "   - 同一主体存在**相互冲突的陈述、行为或身份**（如“我是他朋友，但我必须揭发他”）",
                    "   - 多方之间存在**利益对立**、**期望错位**或**隐藏对抗**（如“表面合作，实则互相防备”）。",

                    "3. **操控行为或意图**：",
                    "   - 使用**情感绑架**（“你不帮我就是不爱我”）、**制造愧疚/恐惧**、**选择性信息披露**、**话术诱导**等手段影响他人。",
                    "   - 描述他人正在被**系统性引导至不利位置**。",

                    "4. **存在可干预的行动窗口**：",
                    "   - 场景中存在一个**关键决策点**、**误解可澄清**、**关系可修复/破坏**、**信息差可填补**，",
                    "   - 且该干预能**实质性改变后续走向**（非纯理论探讨）。",

                    "### 关键排除情形（eligible = false）",
                    "- 仅为情绪宣泄（如“我好难过”、“气死我了”）",
                    "- 仅为事实陈述（如“他昨天去了医院”）",
                    "- 仅为抽象讨论（如“人性本善吗？”、“权力如何腐蚀人？”）",
                    "- 仅为单方面抱怨而无互动结构（如“老板太苛刻了”但无具体事件或对方反应）",
                    "- 仅为求助但无策略/矛盾/操控要素（如“我该怎么办？”但上下文无高阶线索）",

                    "### 判断原则",
                    "- **严格基于文本**：不得引入外部知识或深层隐喻解读。",
                    "- **最小必要推理**：仅允许基于常见语言惯例（如“为了...所以...”表目的，“嘴上...其实...”表矛盾）进行直接推断。",
                    "- **宁缺毋滥**：若无法明确归入上述四类，一律判定为 false。"
                ],
                "fields": {
                    "eligibility": {
                        "type": "object",
                        "properties": {
                            "eligible": {"type": "bool"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 并行步骤
            # 时间感知
            {
                "step_name": LLM_PERCEPTION_TEMPORAL_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 3,
                "label": "感知层：大模型时间感知提取",
                "role": "你是一个严格遵循结构契约的时间信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "temporal",
                "constraint_profile": "high_fidelity_temporal_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效时间事件数据，temporal 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何时间事件，必须输出 `{"temporal": {}}`。',
                "step_rules": [
                    "### 时间事件判定",
                    "- 一个有效的时间事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `temporal_mentions.type` 枚举定义的**显式时间短语**；",
                    "  • 所有 `temporal_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效时间事件**：",
                    "    - 模糊时间词（如“最近”“以前”“将来”）；",
                    "    - 心理时间（如“感觉过了很久”）；",
                    "    - 纯推测（如“可能下周”若未被 uncertain 类型覆盖）；",
                    "    - 隐喻时间（如“时光飞逝”）；",
                    "    - 未被枚举覆盖的孤立时间词。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：提取 1–3 个**连续子字符串**，优先时间动词（如“持续”“等到”）、时间结构（如“从周一到周五”）；**禁止提取非时间性成分**。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当时间谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`visual_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### temporal_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式时间短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `frequency`   → 周期性（“每天”“每周一”）",
                    "  2. `range`       → 区间（“3月到5月”“2020–2023”）",
                    "  3. `relative`    → 相对（“之前”“三天后”“刚离开”）",
                    "  4. `cultural`    → 文化/制度时间（“春节”“周末”“Q3”）",
                    "  5. `duration`    → 持续（“持续两小时”“长达一周”）",
                    "  6. `absolute`    → 具体数值+单位（“2025年3月1日”“两小时”）",
                    "  7. `negated`     → 被否定（“没等到明天”“非工作日”）",
                    "  8. `uncertain`   → 被模糊修饰（“大约下周”“约三小时”）",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效时间短语 → **省略 `temporal_mentions` 字段**。",
                    
                    "### 全局字段",
                    "- `temporal.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `temporal.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                  ],
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
                                        "event_markers": {"type": "array"},
                                        "temporal_mentions": {
                                            "type": "array",
                                            "items": {
                                              "type": "object",
                                              "properties": {
                                                "phrase": {"type": "string"},
                                                "type": {"type": "string"}
                                              }
                                          }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 空间感知
            {
                "step_name": LLM_PERCEPTION_SPATIAL_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 4,
                "label": "感知层：大模型空间感知提取",
                "role": "你是一个严格遵循结构契约的空间信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "spatial",
                "constraint_profile": "high_fidelity_spatial_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效空间事件数据，spatial 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何空间事件，必须输出 `{"spatial": {}}`。',
                "step_rules": [
                    "### 空间事件判定",
                    "- 一个有效的空间事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `spatial_mentions.type` 枚举定义的**显式空间短语**；",
                    "  • 所有 `spatial_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效空间事件**：",
                    "    - 泛称/模糊指代（如“那里”“某个地方”“这边”）；",
                    "    - 心理/隐喻空间（如“心里有个角落”“陷入深渊”）；",
                    "    - 纯动作无空间（如“走过去”若未带方位词）；",
                    "    - 未被枚举覆盖的孤立名词（如单独“桌子”“北京”若非 location 类型上下文）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从 `evidence` 中提取 1 至 3 个最能表征空间关系的**连续子字符串**，优先选择：",
                    "  - 空间介词结构（如“在阳台边”“朝东”）；"
                    "  - 方位动词短语（如“面向大海”“穿过走廊”）；"
                    "  - 空间排布描述（如“一字排开”“坐北朝南”）；"
                    "  - **禁止**提取纯实体名词（如“北京”）、动词（如“走”）或无空间语义的成分。"
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当空间谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`spatial_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### spatial_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式空间短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `relative`    → 相对方位（如“左边”“对面”“附近”）；",
                    "  2. `direction`   → 绝对方向（如“东”“上游”“正南方”）；",
                    "  3. `topological` → 拓扑位置（如“里”“上”“之间”“穿过”）；",
                    "  4. `toward`      → 朝向标记（如“朝东”“面向窗户”“往北走”）；",
                    "  5. `cultural`    → 制度/功能空间（如“VIP区”“后台”“景区入口”）；",
                    "  6. `negated`     → 被否定的空间描述（如“不在屋内”“非公共区域”）；",
                    "  7. `measure`     → 数值+单位的空间量（如“宽3米”“50平方米”）；",
                    "  8. `location`    → 具名地理/建筑实体（如“北京”“三楼会议室”“天安门广场”）；",
                    "  9. `layout`      → 空间排布结构（如“L型”“环形分布”“坐北朝南”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效空间短语 → **省略 `spatial_mentions` 字段**。",

                    "### 全局字段",
                    "- `spatial.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `spatial.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
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
                                        "event_markers": {"type": "array"},
                                        "spatial_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": { "type": "string" },
                                                    "type": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"},
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 视觉感知
            {
                "step_name": LLM_PERCEPTION_VISUAL_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 5,
                "label": "感知层：大模型视觉感知提取",
                "role": "你是一个严格遵循结构契约的视觉信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "visual",
                "constraint_profile": "high_fidelity_visual_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效视觉事件数据，visual 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何视觉事件，必须输出 `{"visual": {}}`。',
                "step_rules": [
                    "### 视觉事件判定",
                    "- 一个有效的视觉事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `visual_mentions.type` 枚举定义的**显式视觉短语**；",
                    "  • 所有 `visual_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效视觉事件**：",
                    "    - 未被枚举覆盖的孤立名词（如“桌子”“红色”作为独立词）；",
                    "    - 纯心理/情感描述（如“她难过”“心里一紧”）；",
                    "    - 隐喻/比喻（如“像霜”“如画”）；",
                    "    - 其他感官模态（如“声音颤抖”“闻到花香”）；",
                    "    - 抽象意图或因果（如“为了你”“因为愧疚”）；",
                    "    - 主观评价（如“看起来危险”“美极了”）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从 `evidence` 中提取 1 至 3 个最能表征视觉动作或状态的**连续子字符串**，优先选择：",
                    "  • 视觉动词（如“望着”“凝视”）；",
                    "  • 动宾结构（如“坐在藤椅上”）；",
                    "  • 姿态短语（如“睫毛低垂”）；",
                    "  • **禁止**提取纯名词、形容词、副词或非动作性成分。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当视觉谓词的主语或宾语**，",
                    "  • 才可将其作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`visual_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### visual_mentions 提取规则（互斥、字面驱动）",
                    "- 所有符合视觉类型的显式短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `color`      → 显式颜色词（如“白色”“灰蓝”）；",
                    "  2. `brightness` → 明暗描述（如“昏暗”“刺眼”）；",
                    "  3. `expression` → 面部表情（如“微笑”“皱眉”）；",
                    "  4. `posture`    → 身体/肢体/头部姿态或可见动作（如“坐着”“低垂”“摩挲”）；",
                    "  5. `object`     → 无生命物 + 视觉修饰（如“旧藤椅”“玻璃门框”）；**孤立名词不提取**；",
                    "  6. `entity`     → 人/动物 + 视觉动词共现（如“狗盯着门”）；",
                    "  7. `optical`    → 光学特性（如“透明”“模糊”）；",
                    "  8. `negated`    → 被“不/没/未/无”直接否定的视觉状态（如“没回头”）；",
                    "  9. `gaze`       → 注视动词+目标（如“望着阳台边的女儿”）；",
                    "  10. `contact`   → 双方视线交互（如“对视”）；",
                    "  11. `medium`    → 视觉媒介（如“透过纱帘”）；",
                    "  12. `occlusion` → 视线遮挡（如“被挡住”）；",
                    "  13. `lighting`  → 环境光照（如“月光下”“灯光昏黄”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效视觉短语 → **省略 `visual_mentions` 字段**。",

                    "### 全局字段",
                    "- `visual.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `visual.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "visual": {
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
                                        "event_markers": {"type": "array"},
                                        "visual_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": {"type": "string"},
                                                    "type": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 听觉感知
            {
                "step_name": LLM_PERCEPTION_AUDITORY_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 6,
                "label": "感知层：大模型听觉感知提取",
                "role": "你是一个严格遵循结构契约的听觉信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "auditory",
                "constraint_profile": "high_fidelity_auditory_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效听觉事件数据，auditory 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何听觉事件，必须输出 `{"auditory": {}}`。',
                "step_rules": [
                    "### 听觉事件判定",
                    "- 一个有效的听觉事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `auditory_mentions.type` 枚举定义的**显式听觉短语**；",
                    "  • 所有 `auditory_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效听觉事件**：",
                    "    - 孤立泛称名词（如“声音”“喊叫”未修饰或无上下文）；",
                    "    - 心理/内源性听觉（如“脑子里有声音”“良心在说话”）；",
                    "    - 隐喻/评价性描述（如“沉默震耳欲聋”“话语如刀”）；",
                    "    - 纯意图或情感投射（如“想听见回应”“希望有人说话”）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从 `evidence` 中提取 1 至 3 个最能表征听觉行为或声学特征的**连续子字符串**，优先选择：",
                    "  - 听觉动词（如“听见”“吼道”）；",
                    "  - 言语内容（如“‘快跑！’”）；",
                    "  - 声音描述（如“咔哒声”“轻声”）；",
                    "  - **禁止**提取纯名词（如“声音”）、无听觉语义的动词（如“说”若未带内容）或评价词（如“刺耳”若未被 intensity/prosody 覆盖）。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当听觉谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`auditory_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### auditory_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式听觉短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `verb`       → 听觉感知或发声动词（如“听见”“喊叫”“窃听”）；",
                    "  2. `speech`     → 可转录的人类言语内容（如“‘快跑！’”“他说会来”）；",
                    "  3. `sound`      → 非语言声音名称或拟声（如“咔哒声”“呜咽”“笑声”）；",
                    "  4. `intensity`  → 声音响度描述（如“大声”“微弱”“震耳欲聋”）；",
                    "  5. `prosody`    → 声音节奏特征（如“急促”“断断续续”“拖长音”）；",
                    "  6. `medium`     → 听觉传播媒介（如“在电话里”“从广播中”）；",
                    "  7. `background` → 环境背景声（如“雨声”“远处车流”）；",
                    "  8. `source`     → 发声主体或物体（如“孩子哭”“门响”）；",
                    "  9. `negated`    → 被“不/没/未/无”直接否定的听觉内容（如“没听见”“无声”）；",
					"- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效时间短语 → **省略 `auditory_mentions` 字段**。",

                    "### 全局字段",
                    "- `auditory.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `auditory.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "auditory": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "experiencer": {"type": "string"},
                                    "evidence": {"type": "array"},
                                    "semantic_notation": {"type": "string"},
                                    "event_markers": {"type": "array"},
                                    "auditory_mentions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "phrase": {"type": "string"},
                                                "type": {"type": "string"}
                                            }
                                        }
                                    },
                                }
                            },
                            "summary": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 嗅觉感知
            {
                "step_name": LLM_PERCEPTION_OLFACTORY_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 7,
                "label": "感知层：大模型嗅觉感知提取",
                "role": "你是一个严格遵循结构契约的嗅觉信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "olfactory",
                "constraint_profile": "high_fidelity_olfactory_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效嗅觉事件数据，olfactory 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何嗅觉事件，必须输出 `{"olfactory": {}}`。',
                "step_rules": [
                    "### 嗅觉事件判定",
                    "- 一个有效的嗅觉事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `olfactory_mentions.type` 枚举定义的**显式嗅觉短语**；",
                    "  • 所有 `olfactory_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效嗅觉事件**：",
                    "    - 孤立泛称表达（如“有味道”“闻了闻”未接具体气味）；",
                    "    - 心理/主观判断无描述（如“他觉得有怪味”但未说明“怪味”是什么）；",
                    "    - 隐喻或抽象用法（如“空气中弥漫着紧张”“爱情的味道”）；",
                    "    - 纯动作无气味内容（如“捂住鼻子”若未提及气味原因）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从 `evidence` 中提取 1 至 3 个最能表征嗅觉行为或气味特征的**连续子字符串**，优先选择：",
                    "  - 气味描述词（如“煤气味”“花香”）；",
                    "  - 嗅觉动作（如“嗅了嗅”“皱鼻”）；",
                    "  - 强度或评价词（如“刺鼻”“清新”）；",
                    "  - ** 禁止 ** 提取纯动词（如“闻”）、无修饰的“味道”、或未绑定气味的反应（如“难受”）。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当嗅觉谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`olfactory_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### olfactory_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式嗅觉短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `odor`      → 具体气味描述（如“花香”“腥味”“煤气味”“臭味”）；",
                    "  2. `source`    → 发出气味的具体实体（如“垃圾”“饭菜”“香水”“尸体”）；",
                    "  3. `intensity` → 气味强度修饰（如“刺鼻”“淡淡”“浓烈”）；",
                    "  4. `valence`   → 感官/情感评价（如“好闻”“难闻”“恶心”“清新”）；",
                    "  5. `negated`   → 被“没/没有/未/无”直接否定的嗅觉内容（如“没闻到”“无异味”）；",
                    "  6. `action`    → 嗅觉动作或生理反应（如“嗅了嗅”“抽鼻子”“捂住鼻子”）；",
                    "  7. `category`  → 气味抽象类别（如“食物”“化学品”“植物”“动物”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效嗅觉短语 → **省略 `olfactory_mentions` 字段**。",

                    "### 全局字段",
                    "- `olfactory.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `olfactory.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "olfactory": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "experiencer": { "type": "string"},
                                        "evidence": { "type": "array"},
                                        "semantic_notation": { "type": "string"},
                                        "event_markers": { "type": "array"},
                                        "olfactory_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": { "type": "string"},
                                                    "type": { "type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": { "type": "string"}
                        }
                      }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 触觉感知
            {
                "step_name": LLM_PERCEPTION_TACTILE_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 8,
                "label": "感知层：大模型触觉感知提取",
                "role": "你是一个严格遵循结构契约的触觉信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "tactile",
                "constraint_profile": "high_fidelity_tactile_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效触觉事件数据，tactile 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何触觉事件，必须输出 `{"tactile": {}}`。',
                "step_rules": [
                    "### 触觉事件判定",
                    "- 一个有效的触觉事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `tactile_mentions.type` 枚举定义的**显式触觉短语**；",
                    "  • 所有 `tactile_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效触觉事件**：",
                    "    - 孤立动作动词无具体触感（如“碰了一下”“摸了摸”未接对象或描述）；",
                    "    - 仅含感知动词无修饰（如“感到”“觉得”后无触觉形容词或名词）；",
                    "    - 心理感受或隐喻（如“心里一紧”“压力山大”）；",
                    "    - 纯意图或假设（如“想摸一下”“可能会疼”）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从`evidence`中提取1至3个最能表征触觉体验的 ** 连续子字符串 **，优先选择：",
                    "  - 触觉形容词（如“粗糙”“滚烫”“刺痛”）；",
                    "  - 身体部位 + 感受组合（如“指尖麻木”“脚底酸胀”）；",
                    "  - 接触方式或动态（如“紧握”“滑动”“震动”）；",
                    "  - ** 禁止 ** 提取纯动词（如“碰”“摸”）、无修饰的“感觉”、或未绑定触觉属性的名词。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当触觉谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`tactile_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### tactile_mentions 字段提取规则（互斥、字面驱动）",
                    "- 所有显式触觉短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `pain`        → 物理性疼痛或异常体感（如“刺痛”“麻木”“酸胀”）；",
                    "  2. `target`      → 被接触的外部物体或表面（如“刀刃”“沙子”“布料”）；",
                    "  3. `body`        → 触觉发生的身体部位（如“指尖”“脚底”“皮肤”）；",
                    "  4. `descriptor`  → 描述体表触感的形容词（如“刺痒”“柔软”“冰冷”）；",
                    "  5. `intensity`   → 触觉强度修饰（如“剧烈”“隐隐”“猛烈”）；",
                    "  6. `texture`     → 表面质地（如“光滑”“毛糙”“粘腻”）；",
                    "  7. `temperature` → 温度感知（如“滚烫”“冰凉”“温热”）；",
                    "  8. `motion`      → 微观动态触感（如“滑动”“摩擦”“拉扯”）；",
                    "  9. `vibration`   → 震动类描述（如“嗡嗡震动”“震颤”）；",
                    " 10. `moisture`    → 湿度状态（如“潮湿”“干燥”“汗湿”）；",
                    " 11. `mode`        → 接触方式（如“贴着”“隔着”“紧握”）；",
                    " 12. `negated`     → 被“没/没有/未/无”直接否定的触觉内容（如“无痛感”“未感到震动”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效触觉短语 → **省略 `tactile_mentions` 字段**。",

                    "### 全局字段",
                    "- `tactile.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `tactile.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "tactile": {
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
                                        "event_markers": {"type": "array"},
                                        "tactile_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": {"type": "string"},
                                                    "type": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                      }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 味觉感知
            {
                "step_name": LLM_PERCEPTION_GUSTATORY_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 9,
                "label": "感知层：大模型味觉感知提取",
                "role": "你是一个严格遵循结构契约的味觉信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "gustatory",
                "constraint_profile": "high_fidelity_gustatory_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效味觉事件数据，gustatory 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何味觉事件，必须输出 `{"gustatory": {}}`。',
                "step_rules": [
                    "### 味觉事件判定",
                    "- 一个有效的味觉事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `gustatory_mentions.type` 枚举定义的**显式味觉短语**；",
                    "  • 所有 `gustatory_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效味觉事件**：",
                    "    - 孤立品尝动作无味觉描述（如“尝了一口”“吃了点东西”未接味道）；",
                    "    - 仅含感知动词无修饰（如“觉得好吃”但未说明“好吃”指什么味）；",
                    "    - 心理评价无具体味觉词（如“这饭不对劲”但未提“苦”“馊”等）；",
                    "    - 隐喻或抽象用法（如“生活的苦”“爱情的甜”）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从`evidence`中提取1至3个最能表征味觉体验的 ** 连续子字符串 **，优先选择：",
                    "  - 基本或复合味觉词（如“苦”“金属味”“油腻”）；",
                    "  - 食物 + 味道组合（如“咖啡很苦”“西瓜甜”）；",
                    "  - 温度或强度修饰（如“烫嘴”“淡淡甜”）；",
                    "  - ** 禁止 ** 提取纯动词（如“尝”“吃”）、无修饰的“味道”、或未绑定味觉属性的评价词（如“好吃”若未搭配具体味型）。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当味觉谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`gustatory_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### gustatory_mentions 字段提取规则（互斥、字面驱动）",
                    "- 所有显式味觉短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `source`      → 被品尝的具体食物/饮品/物质（如“咖啡”“药片”“西瓜”）；",
                    "  2. `basic`       → 基本味觉（如“甜”“苦”“酸”“辣”“咸”“鲜”）；",
                    "  3. `complex`     → 复合或异常口感（如“涩”“油腻”“金属味”“粉感”）；",
                    "  4. `thermal`     → 温度感受（如“冰凉”“烫嘴”“温热”）；",
                    "  5. `evaluation`  → 感官评价（如“美味”“难吃”“怪味”“可口”）；",
                    "  6. `intensity`   → 味道强度修饰（如“极其苦”“微咸”“浓烈辛辣”）；",
                    "  7. `body`        → 味觉相关身体部位（如“舌尖”“喉咙”“口腔”）；",
                    "  8. `negated`     → 被“没/没有/未/无”直接否定的味觉内容（如“无味”“没尝到甜”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效味觉短语 → **省略 `gustatory_mentions` 字段**。",

                    "### 全局字段",
                    "- `gustatory.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `gustatory.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "gustatory": {
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
                                        "event_markers": {"type": "array"},
                                        "gustatory_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": { "type": "string" },
                                                    "type": { "type": "string" }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                      }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 内感受
            {
                "step_name": LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 10,
                "label": "感知层：大模型内感受感知提取",
                "role": "你是一个严格遵循结构契约的内感受信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "interoceptive",
                "constraint_profile": "high_fidelity_interoceptive_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效内感受事件数据，interoceptive 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何内感受事件，必须输出 `{"interoceptive": {}}`。',
                "step_rules": [
                    "### 内感受事件判定",
                    "- 一个有效的内感受事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `interoceptive_mentions.type` 枚举定义的**显式内感受短语**；",
                    "  • 所有 `interoceptive_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效内感受事件**：",
                    "    - 纯情绪词无身体锚点（如“害怕”“焦虑”“紧张”未伴随生理描述）；",
                    "    - 心理隐喻无字面生理词（如“心里一沉”若未出现“胸口闷”“胃紧”等）；",
                    "    - 泛化表达（如“不舒服”“感觉不对”但未指明具体部位或症状）；",
                    "    - 外部动作或环境描述（如“他脸色发白”若未提“头晕”“恶心”等主观体感）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从 `evidence` 中提取 1 至 3 个最能表征内感受体验的**连续子字符串**，优先选择：",
                    "  - 生理症状词（如“心悸”“胃痛”“头晕”“手抖”）；",
                    "  - 身体部位 + 感受组合（如“胸口发闷”“腹部胀满”）；",
                    "  - 强度或触发短语（如“剧烈头痛”“一紧张就心慌”）；",
                    "  - **禁止**提取纯情绪词（如“害怕”）、无修饰的“感觉”、或未绑定生理状态的心理动词。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当内感受谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`interoceptive_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### interoceptive_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式内感受短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `body`            → 内部身体部位（如“胸口”“胃里”“头部”）；",
                    "  2. `cardiac`         → 心跳/心区异常（如“心悸”“胸闷”“心跳加速”）；",
                    "  3. `respiratory`     → 呼吸困难（如“喘不上气”“气短”）；",
                    "  4. `gastrointestinal`→ 胃肠不适（如“胃痛”“反酸”“肠鸣”）；",
                    "  5. `thermal`         → 体温异常（如“发热”“冒冷汗”“燥热”）；",
                    "  6. `muscular`        → 肌肉状态（如“腿软”“手抖”“浑身发颤”）；",
                    "  7. `visceral`        → 内脏压迫感（如“心里沉甸甸”“腹部胀”）；",
                    "  8. `dizziness`       → 头晕/失衡（如“天旋地转”“眼发黑”）；",
                    "  9. `nausea`          → 恶心感（如“想吐”“反胃”）；",
                    " 10. `fatigue`         → 全身疲乏（如“虚脱”“精疲力尽”）；",
                    " 11. `thirst_hunger`   → 口渴/饥饿（如“口干舌燥”“饥肠辘辘”）；",
                    " 12. `intensity`       → 强度修饰（如“隐隐作痛”“剧烈颤抖”）；",
                    " 13. `initiator`       → 触发情境（如“看到血后”“吵架时”“一紧张就”）；",
                    " 14. `negated`         → 被“没/没有/未/无”直接否定的内容（如“无头晕”“未感到恶心”）；",
					"- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效内感受短语 → **省略 `interoceptive_mentions` 字段**。",

                    "### 全局字段",
                    "- `interoceptive.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `interoceptive.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "interoceptive": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "experiencer": { "type": "string" },
                                        "evidence": { "type": "array"},
                                        "semantic_notation": { "type": "string" },
                                        "event_markers": { "type": "array"},
                                        "interoceptive_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": { "type": "string" },
                                                    "type": { "type": "string" }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                      }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 认知过程
            {
                "step_name": LLM_PERCEPTION_COGNITIVE_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 11,
                "label": "感知层：大模型认知过程感知提取",
                "role": "你是一个严格遵循结构契约的认知过程信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "cognitive",
                "constraint_profile": "high_fidelity_cognitive_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效认知过程事件数据，cognitive 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何认知过程事件，必须输出 `{"cognitive": {}}`。',
                "step_rules": [
                    "### 认知事件判定",
                    "- 一个有效的认知事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `cognitive_mentions.type` 枚举定义的**显式认知过程短语**；",
                    "  • 所有 `cognitive_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效认知事件**：",
                    "    - 纯情绪或心理状态无认知动词（如“他很困惑”“显得犹豫”未接“认为”“怀疑”等）；",
                    "    - 仅含表情、动作或副词修饰（如“皱着眉思考”若未出现具体思维内容）；",
                    "    - 隐含意图但无字面表达（如“他走向门口”未说“打算离开”）；",
                    "    - 被动描述他人观点而无主语认知行为（如“据说这药有效”若非“他认为…”）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从`evidence`中提取1至3个最能表征认知活动的 ** 连续子字符串 **，优先选择：",
                    "  - 认知动词 + 宾语结构（如“相信他会来”“怀疑动机”“记得密码”）；",
                    "  - 推理或评价短语（如“推断出真相”“觉得太贵”“意识到错了”）；",
                    "  - ** 禁止 ** 提取纯副词（如“犹豫地”）、无宾语的认知动词（如“他在想”未接内容）、或未绑定具体思维对象的泛化表达。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当认知过程谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`cognitive_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### cognitive_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式认知短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `belief`      → 表达接受为真的信念（如“相信他会来”“认为是真的”）；",
                    "  2. `intention`   → 表达未来行动意图（如“打算离开”“想要尝试”）；",
                    "  3. `inference`   → 基于信息的推理判断（如“推断出真相”“看出破绽”）；",
                    "  4. `memory`      → 对过去的回忆（如“记得那天”“想起她的脸”）；",
                    "  5. `doubt`       → 表达不确定或质疑（如“怀疑动机”“拿不准答案”）；",
                    "  6. `evaluation`  → 对人/事的价值评判（如“觉得太贵”“认为不好”）；",
                    "  7. `solving`     → 针对问题的思考对策（如“想办法解决”“考虑如何应对”）；",
                    "  8. `meta`        → 对自身思维的反思（如“意识到自己错了”“自问为什么”）；",
                    "  9. `certainty`   → 确信程度修饰（如“绝对相信”“隐约觉得”）；",
                    " 10. `negated`     → 被“没/没有/并非/不认为”等直接否定的认知内容（如“不认为可行”“没打算去”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效认知过程短语 → **省略 `cognitive_mentions` 字段**。",

                    "### 全局字段",
                    "- `cognitive.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `cognitive.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "cognitive": {
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
                                      "event_markers": {"type": "array"},
                                      "cognitive_mentions": {
                                          "type": "array",
                                          "items": {
                                              "type": "object",
                                              "properties": {
                                                  "phrase": {"type": "string"},
                                                  "type": {"type": "string"}
                                              }
                                          }
                                      }
                                  }
                              }
                            },
                            "summary": {"type": "string"}
                        }
                      }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 躯体化表现
            {
                "step_name": LLM_PERCEPTION_BODILY_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 12,
                "label": "感知层：大模型躯体化表现感知提取",
                "role": "你是一个严格遵循结构契约的躯体化表现信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "bodily",
                "constraint_profile": "high_fidelity_bodily_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效躯体化表现事件数据，bodily 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何躯体化表现事件，必须输出 `{"bodily": {}}`。',
                "step_rules": [
                    "### 躯体化表现事件判定",
                    "- 一个有效的躯体化事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `bodily_mentions.type` 枚举定义的**显式躯体化表现短语**；",
                    "  • 所有 `bodily_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效躯体化事件**：",
                    "    - 主观内部感受无外显行为（如“感到腿软”若未描述“腿真的发抖”或“站立不稳”）；",
                    "    - 情绪副词或心理状态（如“愤怒地”“吓傻了”“脑子空白”）未伴随可观测动作；",
                    "    - 隐喻、夸张或非字面表达（如“心提到嗓子眼”“像石头一样”）；",
                    "    - 仅提及意图但无实际行为（如“想逃跑”未说“转身就跑”）。",

                    "### events 填充规则",
                    "- `evidence`：必须是从原文中**完整、原样截取**的一个最小句法单元字符串，不得拆分、合并、改写或补全。",
                    "- `event_markers`：从`evidence`中提取1至3个最能表征躯体行为的 ** 连续子字符串 **，优先选择：",
                    "  - 动作动词 + 方向 / 对象（如“冲向门口”“捂住耳朵”）；",
                    "  - 姿态或面部变化（如“弓着背”“咬紧牙关”）；",
                    "  - 自主反应描述（如“手心出汗”“脸色发白”）；",
                    "  - ** 禁止 ** 提取纯情绪副词（如“颤抖地”）、未实现的意图（如“试图站起”若未成功）、或抽象状态词（如“紧张”）。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当躯体化表现谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`bodily_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### bodily_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式躯体化短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `movement`    → 身体整体位移（如“后退”“来回走动”“冲向门口”）；",
                    "  2. `posture`     → 静态姿态（如“蜷缩在角落”“挺直站立”）；",
                    "  3. `facial`      → 面部可见变化（如“皱眉”“瞳孔放大”“咬紧牙关”）；",
                    "  4. `vocal`       → 发声物理异常（如“声音发抖”“说话结巴”）；",
                    "  5. `autonomic`   → 自主神经外显反应（如“手心出汗”“脸红”“浑身发抖”）；",
                    "  6. `freeze`      → 突然静止（如“僵住不动”“愣在原地”）；",
                    "  7. `faint`       → 晕厥前兆（如“眼前发黑”“腿软无力”“要栽倒”）；",
                    "  8. `action`      → 局部目标动作（如“挥手”“踢门”“点头”）；",
                    "  9. `intensity`   → 行为强度修饰（如“剧烈颤抖”“微微点头”）；",
                    "  10. `negated`    → 被“没/没有/未/无”直接否定的行为（如“没有动”“未眨眼”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效躯体化表现短语 → **省略 `bodily_mentions` 字段**。",

                    "### 全局字段",
                    "- `bodily.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `bodily.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "bodily": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "experiencer": {"type": "string"},
                                        "evidence": { "type": "array"},
                                        "semantic_notation": {"type": "string"},
                                        "event_markers": {"type": "array"},
                                        "bodily_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": {"type": "string"},
                                                    "type": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 情感状态
            {
                "step_name": LLM_PERCEPTION_EMOTIONAL_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 13,
                "label": "感知层：大模型情感状态感知提取",
                "role": "你是一个严格遵循结构契约的情感状态信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "emotional",
                "constraint_profile": "high_fidelity_emotional_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效情感状态事件，emotional 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何情感状态事件，必须输出 `{"emotional": {}}`。',
                "step_rules": [
                    "### 情感事件判定",
                    "- 一个有效的情感事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `emotional_mentions.type` 枚举定义的**显式情感状态短语**；",
                    "  • 所有 `emotional_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • 若句中仅含以下内容，则**不构成有效情感事件**：",
                    "    - 纯躯体表现或行为无情绪词共现（如“她哭了”“他颤抖”若未出现“悲伤”“害怕”等）；",
                    "    - 隐喻、比喻或文化习语（如“心碎了”“火冒三丈”若未字面出现标准情绪词）；",
                    "    - 主观推断或外部观察描述（如“看起来很生气”若非“他说自己愤怒”）；",
                    "    - 仅含副词修饰但无核心情绪词（如“愤怒地离开”若“愤怒”未被提取为`adverb`或`emotion`类型——注意： ** “愤怒地”属于`adverb`，有效 **；但“快速地跑开”无情绪词则无效）。",

                    "### events 填充规则",
                    "- evidence 必须是从原文中完整截取的最小句法单元（以句号、问号、感叹号、分号或换行为界）的字符序列,包含合法情感状态表达。",
                    "- `event_markers`：从 `evidence` 中提取 1 至 3 个最能表征情感核心的**连续子字符串**，优先选择：",
                    "  - 情绪词本身（如“焦虑”“happy”）；",
                    "  - 情绪动词+宾语（如“讨厌噪音”“喜欢安静”）；",
                    "  - 情绪副词或形容词（如“愤怒地”“沮丧的”）；",
                    "  - **禁止**提取无情绪语义的通用动词（如“说”“走”）、中性副词（如“很快”“轻轻地”）或未绑定情绪的状态词。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当情感状态谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`emotional_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### emotional_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式情感状态短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. `emotion`     → 具体情绪名词（如“愤怒”“sadness”“jiaolv”“喜悦”）；",
                    "  2. `valence`     → 效价评价词（如“积极”“负面”“good”“不开心”）；",
                    "  3. `arousal`     → 唤醒度状态（如“激动”“平静”“excited”“calm”）；",
                    "  4. `intensity`   → 强度修饰（如“极度”“有点烦”“very”“slightly”）；",
                    "  5. `mode`        → 表达模式线索（如“他感觉”“认为”“noticed that”“自己觉得”）；",
                    "  6. `verb`        → 情绪动词（如“害怕”“hate”“like”“dread”）；",
                    "  7. `adjective`   → 情绪形容词（如“沮丧的”“happy”“annoyed”）；",
                    "  8. `adverb`      → 情绪副词（如“愤怒地”“sadly”“happily”）；",
                    "  9. `mixed`       → 中英混杂表达（如“很 jiaolv”“feel so sad”“非常 happy”）；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效情感状态短语 → **省略 `emotional_mentions` 字段**。",

                    "### 全局字段",
                    "- `emotional.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `emotional.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "emotional": {
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
                                        "event_markers": {"type": "array"},
                                        "emotional_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": {"type": "string"},
                                                    "type": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                      }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 社会关系
            {
                "step_name": LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION,
                "type": PARALLEL_PERCEPTION,
                "index": 14,
                "label": "感知层：大模型社会关系感知提取",
                "role": "你是一个严格遵循结构契约的社会关系信息提取引擎。你的核心原则是：严格依据原文，不错过有效信息，绝不编造未出现的内容。",
                "information_source": "**唯一信源**：所有提取出的属性值必须严格源自 `### USER_INPUT BEGIN` 与 `### USER_INPUT END` 之间的原始文本。禁止使用任何未在上述区块中出现的文本或知识。",
                "driven_by": "social_relation",
                "constraint_profile": "high_fidelity_social_relation_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效社会关系事件数据，social_relation 对象必须包含非空 events 事件列表、evidence 列表、summary 字段。每个 events 事件必须包含非空evidence。",
                    "禁止将多个句子合并为一个 event 的 evidence，除非它们属于同一最小句法单元（如引号内的复合句）。"
                ],
                "empty_result_fallback": '若无任何社会关系事件，必须输出 `{"social_relation": {}}`。',
                "step_rules": [
                    "### 社会关系事件判定",
                    "- 一个有效的社会关系事件必须同时满足：",
                    "  • 其 `evidence` 字段包含且仅包含一个**最小句法单元**（以句号、问号、感叹号、分号或自然换行为界）；",
                    "  • 该句法单元**自身**包含至少一个符合 `social_relation_mentions.type` 枚举定义的**显式社会关系短语**；",
                    "  • 所有 `social_relation_mentions` 中的 `phrase` 必须是该 `evidence` 单元中的**连续子字符串**；",
                    "  • **该句法单元中必须显式出现 ≥2 个可识别的实体或角色指称**（如“张伟和李娜”“老师对我说”“my_brother knows her”），且它们通过关系短语建立连接；",
                    "  • 若句中仅含以下内容，则**不构成有效社会关系事件**：",
                    "    - 单一角色无互动对象（如“他是医生”未提他人）；"
                    "    - 行为描述无关系词（如“他们一起吃饭”若未出现“同事”“朋友”等）；"
                    "    - 隐含关系但无字面表达（如“他递给她文件”未说明身份或关系）；"
                    "    - 跨句拼接的关系（如A句“王芳”，B句“是刘强的妻子” → 若不在同一最小句法单元， ** 无效 **）。"

                    "### events 字段填充规则",
                    "- evidence 必须是从原文中完整截取的最小句法单元（以句号、问号、感叹号、分号或换行为界）的字符序列,包含合法社会关系表达。",
                    "- `event_markers`：从`evidence`中提取1至3个最能表征社会关系核心的 ** 连续子字符串 **，优先选择：",
                    "  - 关系动词（如“认识”“合作”）；",
                    "  - 复合关系词（如“发小”“前夫”）；",
                    "  - 称谓 + 角色组合（如“我的老师”“her_boss”）；",
                    "  - ** 禁止 ** 提取无关系语义的通用动词（如“说”“走”）、孤立名词（如“医生”未与他人共现）或未绑定双方的泛化描述。",
                    "- `experiencer`：",
                    "  • **仅当**该 `evidence` 单元中**显式出现**一个非代词指称（如专有名词、带修饰的身份标签：“穿白裙的女人”“李医生”），且",
                    "  • 该指称在句法上**直接充当社会关系谓词的主语或宾语**，",
                    "  • 才可作为 `experiencer`；",
                    "  • **严禁使用人称代词**（如“他”“她”“他们”）作为 experiencer，即使上下文可推断；",
                    "  • 若不满足上述条件，**必须省略 `experiencer` 字段（包括键名）**。",
                    "- `semantic_notation`：每个有效事件必须包含此字段；",
                    "  • 格式：`social_relation_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须基于 evidence 中明确提及的内容，提炼为一句语法正确、自然流畅的英文短语；全部使用小写字母，禁止包含任何中文字符或标点；允许使用合理的动词形式、介词和连接词以确保语义完整。",

                    "### social_relation_mentions 提取规则（互斥、字面驱动）",
                    "- 所有显式社会关系短语必须提取为 `{'phrase': '原文连续子串', 'type': '单标签'}`；",
                    "- `type` 按以下**严格优先级顺序匹配，首次命中即停止**：",
                    "  1. kinship      → 亲属称谓（含领属修饰），如：妈妈、my_brother、他的父亲、我舅舅；",
                    "  2. role         → 社会身份或职业角色，如：老师、同事、doctor、保安；",
                    "  3. address      → 用于直接称呼的称谓（常出现在对话开头或带语气词），如：老张、hey、您、小李啊；",
                    "  4. possessive   → 非亲属的领属+名词结构，表达社会关联，如：我的朋友、her_boss、他们的邻居；",
                    "  5. relation_verb→ 表示关系建立或互动的动词，如：认识、交往、合作、共事、熟识；",
                    "  6. compound     → 固定复合社会关系词，如：发小、前夫、同班同学、闺蜜、死党、校友；",
                    "  7. duration     → 关系或互动持续时间，如：聊了三小时、认识十年、合作多年；",
                    "  8. distance     → 社交亲疏程度标记，如：不熟、很亲近、barely_know、形同陌路；",
                    "- 每个短语仅提取一次，分配唯一 type；",
                    "- 无有效社会关系短语 → **省略 `social_relation_mentions` 字段**。",

                    "### 全局字段",
                    "- `social_relation.summary`：",
                    "  • 必须是 ≤100 字的**自然中文陈述句**；",
                    "  • **完全基于 `social_relation.evidence` 中实际出现的信息提炼**；",
                    "  • 禁止引入任何未在 evidence 中显式现的信息或逻辑。",
                ],
                "fields": {
                    "social_relation": {
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
                                        "event_markers": {"type": "array"},
                                        "social_relation_mentions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "phrase": {"type": "string"},
                                                    "type": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "summary": {"type": "string"}
                        }
                      }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 其他的串行步骤
            # 策略锚定
            {
                "step_name": LLM_STRATEGY_ANCHOR,
                "type": PARALLEL_HIGH_ORDER,
                "index": 15,
                "label": "策略锚定层：大模型识别行为背后的显性/隐性目标及所利用的社会脚本",
                "role": "你是一个严格受限的行为策略分析引擎。",
                "information_source": (
                    "**唯一信源**：所有提取必须严格源自以下内容："
                    "(1) ### PERCEPTUAL_CONTEXT_BATCH BEGIN/END：已验证的感知事件；"
                    "(2) ### LEGITIMATE_PARTICIPANTS BEGIN/END：合法行为归属主体列表；"
                ),
                "driven_by": "strategy_anchor",
                "constraint_profile": "high_fidelity_strategy_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效策略锚定事件数据，strategy_anchor 对象必须包含非空 events 列表、evidence 列表、synthesis 字段。",
                    "每个 events 事件必须包含非空 behavior、implicit_goal 和 anchor_perceptions。"
                ],
                "empty_result_fallback": '若无任何策略锚定事件，必须输出 `{"strategy_anchor": {}}`。',
                "step_rules": [
                    "### 策略锚定事件判定",
                    "- 一个有效的策略锚定事件需满足以下全部条件：",
                    "  • **行为有据**：behavior 须基于 ≥1 个 PERCEPTUAL_CONTEXT_BATCH 感知事件的 evidence 合理导出；",
                    "  • **主体合法**：agent 必须字面等于 LEGITIMATE_PARTICIPANTS 中的某一指称，且该指称在所引用的 anchor_perceptions 对应的感知事件中明确作为行为主体出现；",
                    "  • **隐性目标可证伪**：implicit_goal 必须能从所引 evidence 中通过行为模式、语言框架或关系不对称直接推得，且若移除该目标，行为就失去策略性解释。",
                    "  • **证据全绑定**：anchor_perceptions 必须包含所有用于推导 behavior 或 implicit_goal 的感知事件 semantic_notation，evidence 必须按相同顺序列出其原始文本；",
                    "  • **禁止孤立生成**：单个感知事件不足以支撑策略锚定；必须基于 ≥2 个感知事件，或单个事件中**同时包含意图声明与冲突行为**。",

                    "### events 填充规则",
                    "- agent：必须字面等于 LEGITIMATE_PARTICIPANTS 中的指称，且该指称在对应 evidence 中明确作为主动施事者出现；禁止使用人称代词；",
                    "- target：若行为有明确作用对象，必须为 LEGITIMATE_PARTICIPANTS 中的指称；否则可省略；",
                    "- behavior：用 ≤20 字精炼概括关键策略性行为，须忠实反映 evidence 的行为实质",
                    "- explicit_justification：若原文中agent 明确声明表面理由（如‘为了你好’‘只是补偿’），则原样提取；否则省略。",
                    "- implicit_goal：必须是从 evidence 中通过逻辑必要性或唯一合理解释反推出的隐性意图（如‘建立道德高位’），若移除该目标则行为失去策略性解释；",
                    "- social_script：若行为明显调用文化/道德脚本（如‘孝道’‘牺牲叙事’‘职场恩赐’），则填写；否则省略；",
                    "- power_differential：若上下文揭示权力/地位/依赖差异（如年龄、职位、经济控制），则简要说明；否则省略；",
                    "- audience_role：若第三方观众被策略性利用（如见证、施压、合法性背书），则说明其功能；否则省略；",
                    "- anchor_perceptions：非空数组，每个元素必须是 PERCEPTUAL_CONTEXT_BATCH 中某感知事件的 semantic_notation；",
                    "- evidence：非空数组，每个元素必须是上述 anchor_perceptions 对应感知事件的原始 evidence 文本，保持相同顺序。",
                    "- semantic_notation：每个有效事件必须包含此字段；",
                    "  • 格式：`strategy_anchor_{english_summary}`（全小写 snake_case，≤128 字符）；",
                    "  • english_summary 必须是从 evidence 提炼出的自然、小写英文短语，不含中文、标点或主观评价，允许使用动词-ing 或介词结构以确保语义完整。",

                    "### 全局字段规则",
                    "- strategy_anchor.synthesis：必须输出 ≤50 字的穿透性策略研判（如‘以道德叙事掩盖权力操控’），直接揭示隐性目标的系统性意图，禁止平铺直叙或事件复述：",
                    "  • 不得罗列事件，必须跨事件抽象；",
                    "  • 禁止引入未在 events 中出现的新概念；",
                    "  • 允许使用精准心理/社会学术语（如‘情感勒索’‘声誉杠杆’）。",
                ],
                "fields": {
                    "strategy_anchor": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "agent": {"type": "string"},
                                        "target": {"type": "string"},
                                        "explicit_justification": {"type": "string"},
                                        "implicit_goal": {"type": "string"},
                                        "behavior": {"type": "string"},
                                        "social_script": {"type": "string"},
                                        "power_differential": {"type": "string"},
                                        "audience_role": {"type": "string"},
                                        "anchor_perceptions": {"type": "array"},
                                        "evidence": {"type": "array"},
                                        "semantic_notation": {"type": "string"}
                                    }
                                }
                            },
                            "synthesis": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 矛盾暴露
            {
                "step_name": LLM_CONTRADICTION_MAP,
                "type": PARALLEL_HIGH_ORDER,
                "index": 16,
                "label": "矛盾暴露层：大模型识别言行不一致、手段-目的错配或情境-行为冲突",
                "role": "你是一个严格受限的逻辑一致性检测引擎。",
                "information_source": (
                    "**唯一信源**：所有提取必须严格源自以下内容："
                    "(1) ### PERCEPTUAL_CONTEXT_BATCH BEGIN/END：已验证的感知事件；"
                    "(2) ### LEGITIMATE_PARTICIPANTS BEGIN/END：合法行为归属主体列表；"
                ),
                "driven_by": "contradiction_map",
                "constraint_profile": "high_fidelity_contradiction_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效矛盾暴露事件，contradiction_map 对象必须包含非空 events 列表、evidence 列表、synthesis 字段。",
                    "每个 events 事件必须包含非空 claimed_premise、actual_behavior、contradiction_type 和 anchor_perceptions。"
                ],
                "empty_result_fallback": '若无任何有效矛盾暴露事件，必须输出 `{"contradiction_map": {}}`。',
                "step_rules": [
                    "### 矛盾暴露事件判定",
                    "- 一个有效的矛盾暴露事件需满足以下全部条件：",
                    "  • **前提有据**：claimed_premise 须基于 ≥1 个 PERCEPTUAL_CONTEXT_BATCH 感知事件的 evidence 合理导出；",
                    "  • **行为有据**：actual_behavior 须基于 ≥1 个 PERCEPTUAL_CONTEXT_BATCH 感知事件的 evidence 合理导出；",
                    "  • **主体一致**：claimed_premise 与 actual_behavior 必须归属于同一 LEGITIMATE_PARTICIPANTS 主体，且该主体在所引感知事件中明确作为言说者或行为主体出现；",
                    "  • **冲突可证伪**：两者之间必须存在逻辑或规范层面的不可调和冲突，且若移除任一要素，冲突即消失；",
                    "  • **证据全绑定**：anchor_perceptions 必须包含所有用于支撑 claimed_premise 或 actual_behavior 的感知事件 semantic_notation，evidence 必须按相同顺序列出其原始文本；",
                    "  • **禁止孤立生成**：单个感知事件不足以构成矛盾暴露；必须基于 ≥2 个感知事件，或单个事件中**同时包含明确声称与冲突行为**。",

                    "### events 字段填充规则",
                    "- claimed_premise：必须忠实还原主体在语境中所表达或可合理反推出的主张、理由、承诺或价值立场；禁止引入未被言语/行为支持的外部归因。",
                    "- actual_behavior：用 ≤20 字精炼描述与前提冲突的实际行为，必须基于 evidence；",
                    "- contradiction_type：必须且只能为以下四者之一：'means_end_mismatch', 'speech_action_split', 'context_violation', 'value_inconsistency'；",
                    "- logical_conflict：用 ≤30 字陈述该矛盾的核心逻辑断裂（如‘声称保护隐私却公开区别对待’）；",
                    "- anchor_perceptions：必须是非空数组，每个元素为 PERCEPTUAL_CONTEXT_BATCH 中某感知事件的 semantic_notation；",
                    "- evidence：必须完全等于所引 anchor_perceptions 对应的原始文本片段，一一对应；",
                    "- semantic_notation：每个有效事件必须包含此字段；",
                    "  • 格式为 contradiction_{contradiction_type}_{english_summary}（全小写 snake_case，≤128字符）；",
                    "  • english_summary 必须是从 evidence 提炼出的自然、小写英文短语，不含中文、标点或主观评价，允许使用动词-ing 或介词结构以确保语义完整。",

                    "### 全局字段规则",
                    "- contradiction_map.synthesis：必须输出 ≤50 字的穿透性矛盾研判（如‘以关怀之名行控制之实’‘表演性道德掩盖系统性偏袒’），直接揭示矛盾的根本之处，禁止平铺直叙或事件复述：",
                    "  • 不得罗列事件，必须跨事件抽象；",
                    "  • 禁止引入未在 events 中出现的新概念；",
                    "  • 允许使用精准逻辑/伦理术语（如‘双重标准’‘自我证成循环’）。",
                ],
                "fields": {
                    "contradiction_map": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "claimed_premise": {"type": "string"},
                                        "actual_behavior": {"type": "string"},
                                        "contradiction_type": {"type": "string",
                                                               "enum": ["means_end_mismatch", "speech_action_split",
                                                                        "context_violation", "value_inconsistency"]},
                                        "logical_conflict": {"type": "string"},
                                        "anchor_perceptions": {"type": "array"},
                                        "evidence": {"type": "array"},
                                        "semantic_notation": {"type": "string"}
                                    }
                                }
                            },
                            "synthesis": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 操控机制解码
            {
                "step_name": LLM_MANIPULATION_DECODE,
                "type": PARALLEL_HIGH_ORDER,
                "index": 17,
                "label": "操控机制解码层：大模型识别情感操控、道德绑架、选择剥夺等心理控制技术",
                "role": "你是一个严格受限的心理操控机制识别引擎。",
                "information_source": (
                    "**唯一信源**：所有提取必须严格源自以下内容："
                    "(1) ### PERCEPTUAL_CONTEXT_BATCH BEGIN/END：已验证的感知事件；"
                    "(2) ### LEGITIMATE_PARTICIPANTS BEGIN/END：合法行为归属主体列表；"
                ),
                "driven_by": "manipulation_decode",
                "constraint_profile": "high_fidelity_manipulation_extraction_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效操控机制事件数据，manipulation_decode 对象必须包含非空 events 列表、evidence 列表、synthesis 字段。",
                    "每个 events 事件必须包含非空 mechanism_type、technique、leverage_point、intended_effect 和 anchor_perceptions。"
                ],
                "empty_result_fallback": '若无任何有效操控机制事件，必须输出 `{"manipulation_decode": {}}`。',
                "step_rules": [
                    "### 操控机制事件判定",
                    "- 一个有效的操控机制事件需满足以下全部条件：",
                    "  • **技术有据**：technique 须基于 ≥1 个 PERCEPTUAL_CONTEXT_BATCH 感知事件的 evidence 合理导出；",
                    "  • **主体合法**：操控实施者（隐含于 technique）必须字面等于 LEGITIMATE_PARTICIPANTS 中的某一名字，且该名字在所引 anchor_perceptions 对应的感知事件中明确作为行为主体出现；",
                    "  • **机制可证伪**：所选 mechanism_type 必须能从所引 evidence 中通过**行为与后果的不对称、重复施压、或选择限制**直接推得，且该机制是解释 intended_effect 的必要前提；",
                    "  • **证据全绑定**：anchor_perceptions 必须包含所有用于支撑 technique、leverage_point 或 intended_effect 的感知事件 semantic_notation，evidence 必须按相同顺序列出其原始文本；",
                    "  • **禁止孤立生成**：单个感知事件不足以构成操控机制；必须基于 ≥2 个感知事件，或单个事件中**同时包含施压行为与被利用的脆弱点线索**。",

                    "### events 字段填充规则",
                    "- mechanism_type：必须且只能为上述七类之一；'guilt_induction', 'fear_appeal', 'love_bombing', 'choice_elimination', 'reputation_leverage', 'false_generosity', 'shame_availability'",
                    "- technique：用 ≤20 字描述具体实施方式（如‘当众给予特殊待遇’‘以沉默制造焦虑’），必须基于 evidence；",
                    "- leverage_point：说明所利用的心理或社会脆弱点（如‘对公平的敏感’‘害怕被说自私’），必须能从所引 evidence 中通过语言框架、情绪施压或关系不对称合理反推，且若移除所依赖的语境线索，该脆弱点即无法成立。",
                    "- intended_effect：用 ≤20 字描述该操控技术预期达成的心理或行为效果（如‘制造情感负债’‘阻断拒绝可能’），必须能通过所引 evidence 与 mechanism_type 的组合合理反推，且该效果是解释 technique 的必要前提。",
                    "- exit_barrier_created：若行为明显制造了退出障碍（如内疚、舆论风险、资源锁定），则简要说明；否则省略；",
                    "- anchor_perceptions：必须是非空数组，每个元素为 PERCEPTUAL_CONTEXT_BATCH 中某感知事件的 semantic_notation；",
                    "- evidence：必须完全等于所引 anchor_perceptions 对应的原始文本片段，一一对应；",
                    "- semantic_notation：每个有效事件必须包含此字段；",
                    "  • 格式为 manipulation_decode_{english_summary}（全小写 snake_case，≤128字符）；",
                    "  • english_summary 必须是从 evidence 提炼出的自然、小写英文短语，不含中文、标点或主观评价，允许使用动词-ing 或介词结构以确保语义完整。",

                    "### 全局字段规则",
                    "- manipulation_decode.synthesis：必须输出 ≤50 字的穿透性操控机制解码研判（如‘以恩惠包装控制’‘用公共见证实施情感勒索’），直接揭示实施者的真实动因和目标，禁止平铺直叙或事件复述：",
                    "  • 不得罗列事件，必须跨事件抽象；",
                    "  • 禁止引入未在 events 中出现的新概念；",
                    "  • 允许使用精准心理战术术语（如‘煤气灯效应’‘沉没成本陷阱’）。",
                ],
                "fields": {
                    "manipulation_decode": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "mechanism_type": {"type": "string",
                                                           "enum": ["guilt_induction", "fear_appeal", "love_bombing",
                                                                    "choice_elimination", "reputation_leverage",
                                                                    "false_generosity", "shame_availability"]},
                                        "technique": {"type": "string"},
                                        "leverage_point": {"type": "string"},
                                        "intended_effect": {"type": "string"},
                                        "exit_barrier_created": {"type": "string"},
                                        "anchor_perceptions": {"type": "array"},
                                        "evidence": {"type": "array"},
                                        "semantic_notation": {"type": "string"}
                                    },
                                }
                            },
                            "synthesis": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [],
            },

            # 最小可行性建议
            {
                "step_name": LLM_MINIMAL_VIABLE_ADVICE,
                "type": SERIAL_SUGGESTION,
                "index": 18,
                "label": "最小可行性建议层：大模型生成可执行、低风险、精准反制操控机制的行动建议",
                "role": "你是一个严格受限的反制策略生成引擎。",
                "information_source": (
                    "**唯一信源**：所有提取必须严格源自以下内容："
                    "(1) ### STRATEGY_ANCHOR_CONTEXT BEGIN/END：已验证的策略锚定信息；"
                    "(2) ### CONTRADICTION_MAP_CONTEXT BEGIN/END：已验证的矛盾暴露信息；"
                    "(3) ### MANIPULATION_DECODE_CONTEXT BEGIN/END：已验证的操控机制解码信息；"
                    "(4) ### LEGITIMATE_PARTICIPANTS BEGIN/END：合法行为归属主体列表；"
                ),
                "driven_by": "minimal_viable_advice",
                "constraint_profile": "high_fidelity_advice_generation_v1",
                "output_prefix": [
                    "请输出一个 JSON 对象，其结构严格遵循以下 schema 的**数据实例形式**："
                ],
                "output_suffix": [
                    "若存在有效建议数据，minimal_viable_advice 对象必须包含非空 events 列表、evidence 列表、synthesis 字段。",
                    "每个 events 事件必须包含非空 counter_action、targeted_mechanism、expected_disruption 和 anchor_perceptions。"
                ],
                "empty_result_fallback": '若无任何有效建议，必须输出 `{"minimal_viable_advice": {}}`。',
                "step_rules": [
                    "### 最小可行性建议事件判定",
                    "- 一个有效建议事件需满足以下全部条件：",
                    "  • **机制有据**：targeted_mechanism 须基于 ≥1 个前序分析单元（strategy_anchor / contradiction_map / manipulation_decode）的输出合理导出；",
                    "  • **主体一致**：所针对的操控实施者必须字面等于 LEGITIMATE_PARTICIPANTS 中的指称，且在所引 anchor_perceptions 对应的前序事件中明确作为行为主体出现；",
                    "  • **行动可破局**：counter_action 必须能直接中断 targeted_mechanism 的 intended_effect 或 leverage_point，且在原文情境下具备现实可行性；",
                    "  • **证据全绑定**：anchor_perceptions 必须包含所有用于识别 targeted_mechanism 的前序分析单元 semantic_notation，evidence 必须按相同顺序列出其原始文本；",
                    "  • **禁止幻想方案**：不得建议依赖第三方、制度干预、高风险对抗或未在上下文中暗示的资源；宁可无建议，不可越界。",

                    "### events 字段填充规则",
                    "- counter_action：用 ≤25 字描述具体、可独立执行的最小破局动作（如‘当众平静回应：我不需要特殊待遇’），必须基于原文情境可行；",
                    "- targeted_mechanism：明确指向前序分析中的具体机制名称，术语须与前序输出一致；",
                    "- expected_disruption：说明该行动如何瓦解操控链（如‘使恩惠失去强制接受性’‘暴露补偿与隐私保护的逻辑冲突’）；",
                    "- feasibility_condition：若行动依赖特定前提（如‘在场有第三方’‘对方使用原话’），则简要说明；否则省略；",
                    "- anchor_perceptions：非空数组，每个元素必须是前序分析中某单元的 semantic_notation；",
                    "- evidence：必须完全等于所引 anchor_perceptions 对应的原始 evidence 文本，一一对应；",
                    "- semantic_notation：每个有效事件必须包含此字段；",
                    "  • 格式为 minimal_viable_advice_{english_summary}（全小写 snake_case，≤128字符）；",
                    "  • english_summary 必须是从 evidence 提炼出的自然、小写英文短语，不含中文、标点或主观评价，允许使用动词-ing 或介词结构以确保语义完整。",

                    "### 全局字段规则",
                    "- minimal_viable_advice.synthesis：必须是 ≤20 字的穿透性行为策略逆向工程标签，融合策略意图、逻辑矛盾与操控机制，形成复合诊断性命题；",
                    "  • 示例：‘欲盖弥彰+强迫式瓮中捉鳖’‘表演性补偿掩盖控制意图’；",
                    "  • 禁止方法论、行动指南或抽象原则（如‘设立边界’‘保持冷静’）；",
                    "  • 必须可回溯至 events 中的 targeted_mechanism 与 expected_disruption；",
                    "  • 允许使用高密度修辞组合（如‘道德高位绑架+选择剥夺’）。",
                ],
                "fields": {
                    "minimal_viable_advice": {
                        "type": "object",
                        "properties": {
                            "events": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "counter_action": {"type": "string"},
                                        "targeted_mechanism": {"type": "string"},
                                        "expected_disruption": {"type": "string"},
                                        "feasibility_condition": {"type": "string"},
                                        "anchor_perceptions": {"type": "array"},
                                        "evidence": {"type": "array"},
                                        "semantic_notation": {"type": "string"}
                                    }
                                }
                            },
                            "synthesis": {"type": "string"}
                        }
                    }
                },
                "version": "v1.0.0",
                "changelog": [
                    "对齐前三层推演风格；synthesis 定位为 ≤20 字复合策略标签；强化机制回溯与破局逻辑"
                ]
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
    ),
    GLOBAL_SEMANTIC_SIGNATURE: """
        你是一个语义摘要器，任务是为以下用户输入生成一个全局语义标识。

        ### 用户输入开始
        {user_input}
        ### 用户输入结束
        
        要求：
        - 输出格式：raw_{{emotion}}_{{english_summary}};
        - 全小写，snake_case，总长度 ≤256 字符;
        - emotion 必须从以下列表中选择最贴切的一个：  
          neutral, joy, sadness, anger, fear, surprise, disgust,  
          shame, guilt, pride, envy, gratitude, hope, despair,  
          anxiety, frustration, confusion, overwhelm, loneliness,  
          regret, resentment, bitterness, melancholy, apprehension,  
          dread, relief, contentment, nostalgia, ambivalence, inexpressible,  
          tension, unease, restlessness, emptiness, numbness, complex
        
        - english_summary 必须：
          • 严格基于用户输入中明确提及或直接暗示的内容提炼为一句自然、流畅、语法正确的英文短语；
          • 转为全小写，空格和标点替换为下划线；
          • 可使用动词、介词等保证语义完整；
        
        - 只输出一行字符串，不要任何其他内容
    """
}
