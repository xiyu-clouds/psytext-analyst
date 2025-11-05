from src.state_of_mind.utils.constants import CATEGORY_RAW, LLM_SOURCE_EXTRACTION, \
    LLM_PERCEPTION_TEMPORAL_EXTRACTION, LLM_PERCEPTION_SPATIAL_EXTRACTION, \
    LLM_PERCEPTION_VISUAL_EXTRACTION, LLM_PERCEPTION_GUSTATORY_EXTRACTION, LLM_PERCEPTION_TACTILE_EXTRACTION, \
    LLM_PERCEPTION_OLFACTORY_EXTRACTION, LLM_PERCEPTION_AUDITORY_EXTRACTION, LLM_PERCEPTION_EMOTIONAL_EXTRACTION, \
    LLM_PERCEPTION_SOCIAL_RELATION_EXTRACTION, SERIAL, PARALLEL, LLM_PERCEPTION_INTEROCEPTIVE_EXTRACTION, \
    LLM_PERCEPTION_COGNITIVE_EXTRACTION, LLM_PERCEPTION_BODILY_EXTRACTION, LLM_INFERENCE, LLM_RATIONAL_ADVICE, \
    LLM_DEEP_ANALYSIS, PREPROCESSING, CATEGORY_SUGGESTION

LLM_PROMPTS_SCHEMA = {
    CATEGORY_RAW: {
        "version": "1.0.0",
        "core_iron_law": """
            ### 0.【对话重置：无视历史，仅响应当前输入】—— 上下文隔离原则
            - 完全忽略此前对话，仅基于本次 `fields` 模板和用户输入文本处理。
            - 禁止模仿、延续或参考任何历史输出。
            
            ### 1.【存在即字段，无据即抹除】—— 字段存在性铁律
            - **可选字段（required=False）**：仅当用户输入中存在与字段描述语义直接对应的显式文本时，才输出该字段；否则**必须彻底省略（不出现字段名，不返回 {}/""/[] 等任何形式）**。
            - **必填字段（required=True）**：无论是否有显式内容，**必须始终存在**，并严格按其类型返回合法值：
              - `string` → `""`（空字符串）
              - `array` → `[]`（空列表）
              - `object` → `{}`（空对象）
            
            ### 2.【字面主义：禁止推断、补全、常识】—— 信息保真原则
            - 所有内容必须有**直接文字依据**，禁止隐含解读、角色补全、情感归因或逻辑延伸。
            
            ### 3.【结构神圣：名、型、层、式四位一体】—— 结构一致性原则
            - 输出 JSON 必须在以下四维与模板完全一致：
              - **字段名**：完全匹配（含嵌套路径），禁止改写。
              - **字段类型**：`string`→字符串；`array`→**列表**；`object`→对象； `number`→数值；`boolean`→布尔类型。
              - **字段格式**：[str]，必须为字符串列表,即使仅提取到一项也必须用 ["..."] 格式，禁止返回裸字符串、null 或非列表类型。
              - **字段层级**：保持嵌套结构，禁止扁平化。
              - 若你决定输出以下`fields`模板结构对象，则必须同时包含以下三个字段：
                - evidence：原文逐字引用，不得改写、总结或推理；
                - semantic_notation：snake_case，≤64字符，无专有名词，无数字，仅描述显式或直接可推的语义模式；
                - summary：≤100字，客观概括，不得引入新信息。
                
            > ⚠️ 数组字段一旦输出，必须为 `[值1, 值2, ...]` 格式；**绝不允许裸字符串或空列表 `[]`。**
            
            ### 4.【纯净输出：仅返回合法 JSON】—— 输出洁净原则
            - 输出必须是语法合法的 JSON 对象，无前缀、后缀、注释、Markdown 或额外字符。
            
            ### 5.【按需触发：字段级原子存在】—— 精简存在原则
            - 字段出现 = 用户输入中有显式语义单元可映射。
            - 数组元素必须是**被明确提及的标准化标签**，不得添加默认值、推测项或占位符。
            - **无提及 → 无字段 → 无痕迹。**
            
            ### 6.【语义原子化：机器可解析优先】—— 数据注入友好原则
            - 字符串值（尤其数组元素）必须为标准化、低歧义标签，优先使用：
              - `snake_case`
              - 动词+宾语结构
              - 心理学/行为学术语
            - 禁止完整句子、文学描写、模糊形容词。
            
            ### 7.【最大捕获：不错过任何显式信息】—— 信息密度最大化原则
            - 在字面主义前提下，穷尽所有可提取的显式语义单元。
            - 允许对复合句解耦，只要每项有文本依据。
            - 目标：**不编造，但不错过。**
        """,
        # === 控制流：处理阶段划分 ===
        "pipeline": [
            {
                "step": LLM_SOURCE_EXTRACTION,
                "type": PREPROCESSING,
                "index": 0,
                "label": "预处理：大模型源信息提取",
                "role": "你是一个严格遵循结构契约的信息提取引擎，必须对每个识别出的参与者执行全文扫描，提取其所有显式关联的属性",
                "sole_mission": "你的唯一任务是从用户输入中识别所有**显式出现的、用于指代特定人类个体的名词短语**（称为“参与者”），并按指定结构输出最小存在、最大保真的JSON",
                "driven_by": "participants",
                "fields": {
                    "participants": {
                        "type": "array",
                        "required": True,
                        "format": "[{entity: str, name: str, social_role: str, age_range: str, gender: str, ethnicity_or_origin: str, physical_traits: [str], appearance: [str], baseline_health: str, inherent_odor: [str], voice_quality: str, affective_orientation: [str], personality_traits: [str], behavioral_tendencies: [str], education_level: str, occupation: str, socioeconomic_status: str, cultural_identity: [str], primary_language: str}]",
                        "description": (
                            "必须是**显式名词短语**，能唯一或典型地指向一个具体人类角色",
                            "必须包含**词汇性内容**（不能是代词如“他”，也不能是泛指如“有人”）",
                            "示例合法 entity: `母亲`，`李医生`，`“站在窗边的女孩”`，`穿黑西装的男人`等",
                            "示例非法 entity：`他`，`他们`，`一个人`，`观众`",
                            "对于每个 `entity`，你必须扫描全文，找出所有**在句法或语篇上直接描述该 entity 的显式片段**，并按语义映射到对应字段",
                            "属性提取必须基于**字面显式描述**，禁止任何推理、常识补充或默认值",
                            "若某字段无显式描述，则该字段必须完全省略（不得为 null、空字符串或空数组）",
                            "例如：输入‘母亲声音发颤’ → entity='母亲', voice_quality='声音发颤'"
                        ),
                        "items": {
                            "entity": {
                                "type": "string",
                                "required": True,
                                "description": "必须完整复制用户输入中的原始指称短语，一字不差"
                            },
                            "name": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中出现具体姓名（如“张伟”）时填写，否则省略"
                            },
                            "social_role": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户明确使用社会角色词（如`医生`，`老师`，`班长`）时填写"
                            },
                            "age_range": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及年龄范围（如`三十多岁`，`青少年`）才提取"
                            },
                            "gender": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及性别身份或表达（如`她是个女孩`，`男性`）才提取"
                            },
                            "ethnicity_or_origin": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及族群、国籍或地域出身（如`来自云南`，`汉族`）才提取"
                            },
                            "physical_traits": {
                                "type": "array",
                                "required": False,
                                "description": "仅提取固有、长期存在的生理特征（如“秃顶”“高个子”“疤痕”）必须为显式描述"
                            },
                            "appearance": {
                                "type": "array",
                                "required": False,
                                "description": "提取外貌、衣着、姿态、动作等视觉可辨的显式描述（如“穿着白衬衫”“低头不语”“手指摩挲门框”）"
                            },
                            "baseline_health": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及基础健康状况或慢性病史（如`患有哮喘`，`身体虚弱`）才提取"
                            },
                            "inherent_odor": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及固有体味或气味特征（如`带着淡淡的药味`，`有汗味`）才提取"
                            },
                            "voice_quality": {
                                "type": "string",
                                "required": False,
                                "description": "提取声音特质的显式描述（如`声音沙哑`，`语速缓慢`，`带着哽咽`，`更低更沉`）"
                            },
                            "affective_orientation": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及情感依恋风格（如`回避型依恋`，`渴望亲密`）才提取"
                            },
                            "personality_traits": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及长期人格特质（如`性格内向`，`脾气急躁`，`温柔体贴`）才提取"
                            },
                            "behavioral_tendencies": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及稳定行为倾向或习惯（如`习惯性咬指甲`，`说话前会停顿`）才提取"
                            },
                            "education_level": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及教育程度（如`硕士毕业`，`没上过学`）才提取"
                            },
                            "occupation": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及职业身份（如`护士`，`自由职业者`）才提取"
                            },
                            "socioeconomic_status": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及社会经济地位（如`家境贫寒`，`中产家庭`）才提取"
                            },
                            "cultural_identity": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及文化身份标签（如`潮汕人`，`基督徒`，`二次元爱好者`）才提取"
                            },
                            "primary_language": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及主要使用语言（如`只会说方言`，`英语流利`）才提取"
                            }
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
                "role": "你是**心海系统**的超级时间感知解析器。",
                "sole_mission": "你的唯一任务是：从用户输入和参与者列表描述中识别与字段模板完全匹配的显式依据，并按指定结构输出**最小存在、最大保真**的JSON。禁止任何推理、补全、常识推断或语义泛化",
                "driven_by": "temporal",
                "fields": {
                    "temporal": {
                        "type": "object",
                        "required": False,
                        "format": "{exact_literals: [str], relative_expressions: [str], reference_anchor: str, time_ranges: [str], durations: [str], frequencies: [str], experiencer: str, evidence: [str], semantic_notation: str, summary: str}",
                        "description": "仅当用户输入中明确包含时间相关描述时才输出该对象。所有子字段必须基于显式文本，禁止推断、补全或语义推演。所有数组字段必须为标准列表格式，哪怕只提取到一项数据，禁止返回裸字符串、null 或非列表类型",
                        "items": {
                            "exact_literals": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及了精确时间字面量才提取"
                            },
                            "relative_expressions": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及了相对或模糊时间表达才提取"
                            },
                            "reference_anchor": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及了用于解析相对时间的参考时间锚点才提取"
                            },
                            "time_ranges": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及了时间区间才提取"
                            },
                            "durations": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及了持续时间表达（如“持续两小时”）才提取"
                            },
                            "frequencies": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及了周期性或频率表达才提取"
                            },
                            "experiencer": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及了时间事件的感知或陈述主体，且该主体已在参与者列表中被提取，才提取此字段"
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 temporal 对象，则此字段必须存在且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 temporal 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 temporal 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级空感知解析器。",
                "sole_mission": "你的唯一任务是：从用户输入和参与者列表描述中识别与字段模板完全匹配的显式描述，并按指定结构输出**最小存在、最大保真**的JSON。禁止任何推理、补全、常识推断或语义泛化",
                "driven_by": "spatial",
                "fields": {
                    "spatial": {
                        "type": "object",
                        "required": False,
                        "format": "{places: [str], layout_descriptions: [str], experiencer: str, proximity_relations: [{actor: str, target: str, distance_cm: int, medium: [str], channel: [str], barrier: [str], relation_type: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "仅当用户输入中明确包含空间相关描述时才输出该对象。所有子字段必须基于显式文本，禁止推断、补全或语义泛化。所有数组字段必须为标准列表格式，哪怕只提取到一项数据，禁止返回裸字符串、null 或非列表类型",
                        "items": {
                            "places": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确提及了具体地点或场所名称才提取"
                            },
                            "layout_descriptions": {
                                "type": "array",
                                "required": False,
                                "description": "仅当用户输入中明确描述了空间结构或布局才提取"
                            },
                            "experiencer": {
                                "type": "string",
                                "required": False,
                                "description": "仅当用户输入中明确提及了空间描述的感知或陈述主体，且该主体已在参与者列表中被提取，才提取此字段"
                            },
                            "proximity_relations": {
                                "type": "array",
                                "required": False,
                                "format": "[{actor: str, target: str, distance_cm: int, medium: [str], channel: [str], barrier: [str], relation_type: str}]",
                                "description": "仅当用户输入中明确描述了两个或多个参与者之间的空间关系时才提取。每个关系对象中的 actor 和 target 必须严格匹配已提取的参与者角色",
                                "items": {
                                    "actor": {
                                        "type": "string",
                                        "required": False,
                                        "description": "仅当用户输入中明确指定了空间关系的主动方，且该角色已在参与者列表中被提取，才提取此字段"
                                    },
                                    "target": {
                                        "type": "string",
                                        "required": False,
                                        "description": "仅当用户输入中明确指定了空间关系的目标方，且该角色已在参与者列表中被提取，才提取此字段"
                                    },
                                    "distance_cm": {
                                        "type": "integer",
                                        "required": False,
                                        "description": "仅当用户输入中明确给出了数值距离（含单位）才提取，并统一换算为厘米（cm）。无显式依据时，必须彻底省略该字段"
                                    },
                                    "medium": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确提及了信息或互动所依赖的物理/感知媒介才提取"
                                    },
                                    "channel": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确描述了互动所使用的渠道或方式才提取"
                                    },
                                    "barrier": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确指出了阻碍感知或移动的障碍物才提取"
                                    },
                                    "relation_type": {
                                        "type": "string",
                                        "required": False,
                                        "description": "仅当用户输入中明确使用了空间关系类型词才提取"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 spatial 对象，则此字段必须存在且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 spatial 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 spatial 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级视觉感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式视觉描述。"
                    "仅当某角色在原文中有明确视觉行为时，才为其生成一个 visual 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "visual",
                "fields": {
                    "visual": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, observed_entity: str, visual_objects: [str], visual_attributes: [str], visual_actions: [str], gaze_target: str, eye_contact: [str], facial_cues: [str], salience: float, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "visual 对象仅在用户输入显式描述了 participants 中某角色的视觉行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, observed_entity: str, visual_objects: [str], visual_attributes: [str], visual_actions: [str], gaze_target: str, eye_contact: [str], facial_cues: [str], salience: float, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的视觉行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "仅当用户输入中明确提及了观察主体，且该主体已在 participants 列表中才提取此字段"
                                    },
                                    "observed_entity": {
                                        "type": "string",
                                        "required": False,
                                        "description": "仅当用户输入中明确提及了被观察的对象或主体，且该对象已在 participants 列表中才提取此字段"
                                    },
                                    "visual_objects": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确提及了可见物体才提取"
                                    },
                                    "visual_attributes": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确描述了对象的视觉属性才提取"
                                    },
                                    "visual_actions": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确描述了可见的动作或姿态才提取"
                                    },
                                    "gaze_target": {
                                        "type": "string",
                                        "required": False,
                                        "description": "仅当用户输入中明确指出注视目标才提取"
                                    },
                                    "eye_contact": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确描述了眼神交互才提取"
                                    },
                                    "facial_cues": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中明确提及了面部表情或微表情线索才提取"
                                    },
                                    "salience": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当用户输入中明确包含视觉显著性或确定性修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前视觉事件提供直接支持的原文片段。仅包含用户输入中明确描述该事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前视觉事件生成一个高度提炼的语义标识符，仅反映该事件的核心视觉行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 visual 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 visual 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 visual 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级听觉感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式听觉描述。"
                    "仅当某角色在原文中有明确听觉行为时，才为其生成一个 auditory 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "auditory",
                "fields": {
                    "auditory": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, sound_source: str, auditory_content: [str], is_primary_focus: bool, rhetorical_patterns: [str], prosody_cues: [str], pause_description: str, intensity: float, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "auditory 对象仅在用户输入显式描述了 participants 中某角色的听觉行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, sound_source: str, auditory_content: [str], is_primary_focus: bool, rhetorical_patterns: [str], prosody_cues: [str], pause_description: str, intensity: float, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的听觉行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "听觉接收主体（谁在听），必须严格匹配已提取的 participants 列表中的功能角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "sound_source": {
                                        "type": "string",
                                        "required": False,
                                        "description": "发声主体或声源（谁在说/发出声音），必须严格匹配已提取的 participants 列表中的功能角色"
                                    },
                                    "auditory_content": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接出现的听觉内容原文或关键词，必须为显式描述，禁止抽象语义标签或概括"
                                    },
                                    "is_primary_focus": {
                                        "type": "boolean",
                                        "required": False,
                                        "description": "仅当用户输入中明确表示注意力集中于该声音时才为 true，否则不得输出此字段"
                                    },
                                    "rhetorical_patterns": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中直接使用修辞术语或明显引用修辞结构时才提取"
                                    },
                                    "prosody_cues": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当用户输入中直接描述声音特征时才提取"
                                    },
                                    "pause_description": {
                                        "type": "string",
                                        "required": False,
                                        "description": "仅当用户输入中明确描述停顿时长或程度时才提取，禁止估算或补全"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当用户输入中包含明确听觉强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前听觉事件提供直接支持的原文片段。仅包含用户输入中明确描述该听觉事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前听觉事件生成一个高度提炼的语义标识符，仅反映该事件的核心听觉行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 auditory 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 auditory 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 auditory 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级嗅觉感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式嗅觉描述。"
                    "仅当某角色在原文中有明确嗅觉行为时，才为其生成一个 olfactory 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "olfactory",
                "fields": {
                    "olfactory": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, odor_source: str, odor_descriptors: [str], intensity: float, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "olfactory 对象仅在用户输入显式描述了 participants 中某角色的嗅觉行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, odor_source: str, odor_descriptors: [str], intensity: float, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的嗅觉行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "气味感知主体（谁在闻），必须严格匹配已提取的 participants 列表中的角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "odor_source": {
                                        "type": "string",
                                        "required": False,
                                        "description": "气味来源（谁/什么发出气味），必须为用户输入中明确指出的对象或角色，禁止推测或泛化"
                                    },
                                    "odor_descriptors": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接出现的气味描述词或短语，必须为原文显式表达，禁止抽象语义标签"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当用户输入中包含明确嗅觉强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前嗅觉事件提供直接支持的原文片段。仅包含用户输入中明确描述该嗅觉事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前嗅觉事件生成一个高度提炼的语义标识符，仅反映该事件的核心视嗅觉为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 olfactory 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 olfactory 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 olfactory 对象被输出时，≤100字，客观概括，不得引入新信息"
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
                "role": "你是**心海系统**的超级触觉感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式触觉描述。"
                    "仅当某角色在原文中有明确触觉行为时，才为其生成一个 tactile 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "tactile",
                "fields": {
                    "tactile": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, contact_target: str, tactile_descriptors: [str], intensity: number, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "tactile 对象仅在用户输入显式描述了 participants 中某角色的触觉行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, contact_target: str, tactile_descriptors: [str], intensity: number, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的触觉行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "触觉体验主体（谁感受到触觉），必须严格匹配已提取的 participants 列表中的功能角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "contact_target": {
                                        "type": "string",
                                        "required": False,
                                        "description": "被接触的对象或身体部位，必须为用户输入中明确指出的内容，禁止推测或泛化"
                                    },
                                    "tactile_descriptors": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接描述的触觉感受或动作，必须为原文显式表达，禁止抽象语义标签或概括"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当用户输入中包含明确触觉强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前触觉事件提供直接支持的原文片段。仅包含用户输入中明确描述该触觉事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前触觉事件生成一个高度提炼的语义标识符，仅反映该事件的核心触觉行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 tactile 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 tactile 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 tactile 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级味觉感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式味觉描述。"
                    "仅当某角色在原文中有明确味觉行为时，才为其生成一个 gustatory 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "gustatory",
                "fields": {
                    "gustatory": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, taste_source: str, taste_descriptors: [str], intensity: number, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "gustatory 对象仅在用户输入显式描述了 participants 中某角色的味觉行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, taste_source: str, taste_descriptors: [str], intensity: number, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的味觉行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "味觉体验主体（谁在尝），必须严格匹配已提取的 participants 列表中的功能角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "taste_source": {
                                        "type": "string",
                                        "required": False,
                                        "description": "食物或味道来源，必须为用户输入中明确指出的物理载体，禁止推测未提及的来源"
                                    },
                                    "taste_descriptors": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接出现的味道描述词或短语，必须为原文显式表达，禁止抽象标签或概括性语义"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当用户输入中包含明确味觉强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前味觉事件提供直接支持的原文片段。仅包含用户输入中明确描述该味觉事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前味觉事件生成一个高度提炼的语义标识符，仅反映该事件的核心味觉行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 gustatory 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 gustatory 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 gustatory 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级内感受感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式内感受描述。"
                    "仅当某角色在原文中有明确内感受行为时，才为其生成一个 interoceptive 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "interoceptive",
                "fields": {
                    "interoceptive": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, body_sensation: [str], intensity: number, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "interoceptive 对象仅在用户输入显式描述了 participants 中某角色的内感受行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, body_sensation: [str], intensity: number, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的内感受行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "主观感受的体验者（谁感受到身体内部状态），必须严格匹配已提取的 participants 列表中的功能角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "body_sensation": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接出现的身体内部感觉描述，必须为原文显式表达或标准生理描述，禁止抽象标签（如‘焦虑感’）或情绪推断"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当用户输入中包含明确内感受强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前内感受事件提供直接支持的原文片段。仅包含用户输入中明确描述该内感受事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前内感受事件生成一个高度提炼的语义标识符，仅反映该事件的核心内感受行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 interoceptive 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 interoceptive 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 interoceptive 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
                        }
                    }
                }
            },

            # 认知过程
            {
                "step": LLM_PERCEPTION_COGNITIVE_EXTRACTION,
                "type": PARALLEL,
                "index": 9,
                "label": "感知层：大模型认知过程感知提取",
                "role": "你是**心海系统**的超级认知感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式认知过程描述。"
                    "仅当某角色在原文中有明确认知过程行为时，才为其生成一个 cognitive 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "cognitive",
                "fields": {
                    "cognitive": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, explicit_thought: [str], intensity: number, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "cognitive 对象仅在用户输入显式描述了 participants 中某角色的认知过程行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, explicit_thought: [str], intensity: number, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的认知过程行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "认知主体（谁在思考），必须严格匹配已提取的 participants 列表中的功能角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "explicit_thought": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接表达的思维内容，必须为原文显式陈述。禁止将行为、表情、情绪或隐含意图当作思维内容"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当思维内容中包含明确认知强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前认知过程事件提供直接支持的原文片段。仅包含用户输入中明确描述该认知过程事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前认知过程事件生成一个高度提炼的语义标识符，仅反映该事件的核心认知过程行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 cognitive 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 cognitive 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 cognitive 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级躯体化表现感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式躯体化表现描述。"
                    "仅当某角色在原文中有明确躯体化表现行为时，才为其生成一个 bodily 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "bodily",
                "fields": {
                    "bodily": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, observable_behavior: [str], intensity: number, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "bodily 对象仅在用户输入显式描述了 participants 中某角色的躯体化表现行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, observable_behavior: [str], intensity: number, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的躯体化表现行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "躯体行为主体（谁做出了该行为），必须严格匹配已提取的 participants 列表中的功能角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "observable_behavior": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接描述的外部可观测身体行为，必须为原文显式表达，禁止将情绪标签或意图当作躯体行为"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当行为描述中包含明确强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前躯体化表现事件提供直接支持的原文片段。仅包含用户输入中明确描述该躯体化表现事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前躯体化表现事件生成一个高度提炼的语义标识符，仅反映该事件的核心躯体化表现行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 bodily 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 bodily 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 bodily 对象被输出时，≤100字，客观概括，不得引入新信息"
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
                "role": "你是**心海系统**的超级情感状态感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式情感状态描述。"
                    "仅当某角色在原文中有明确情感状态行为时，才为其生成一个 emotional 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "emotional",
                "fields": {
                    "emotional": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{experiencer: str, emotion_labels: [str], valence: number, arousal: number, intensity: number, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "emotional 对象仅在用户输入显式描述了 participants 中某角色的情感状态行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{experiencer: str, emotion_labels: [str], valence: number, arousal: number, intensity: number, evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的情感状态行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "string",
                                        "required": False,
                                        "description": "情绪表达主体（谁在经历情绪），必须严格匹配已提取的 participants 列表中的功能角色。若用户输入提及主体但该角色未被提取，则不得输出此字段"
                                    },
                                    "emotion_labels": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接出现的情绪词或短语，必须为原文显式表达。禁止将‘他握紧拳头’等行为当作情绪标签"
                                    },
                                    "valence": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": -1.00,
                                        "maximum": 1.00,
                                        "description": "仅当情绪标签本身或上下文包含明确效价词时才可量化。若无显式依据，必须彻底省略该字段"
                                    },
                                    "arousal": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当情绪标签或修饰语包含唤醒度线索时才赋值；若无任何唤醒相关词，必须彻底省略该字段"
                                    },
                                    "intensity": {
                                        "type": "number",
                                        "required": False,
                                        "minimum": 0.00,
                                        "maximum": 1.00,
                                        "description": "仅当情绪表达中包含强度修饰词时才可量化。无显式依据时，必须彻底省略该字段"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前情感状态事件提供直接支持的原文片段。仅包含用户输入中明确描述该情感状态事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前情感状态事件生成一个高度提炼的语义标识符，仅反映该事件的核心情感状态行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 emotional 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 emotional 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 emotional 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
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
                "role": "你是**心海系统**的超级社会关系感知解析器。",
                "sole_mission": (
                    "你的唯一任务是：基于用户输入内容和注入的 participants 列表描述中的每一个角色，"
                    "严格检查用户输入中是否存在**该角色作为观察主体**的显式社会关系描述。"
                    "仅当某角色在原文中有明确社会关系行为时，才为其生成一个 social_relation 对象。"
                    "禁止跳过任何角色，禁止合并多个角色的感知，禁止虚构未提及的观察行为，禁止使用非 participants 中的角色名"
                    "⚠️ 注意：每个 events 条目必须独立包含自己的 evidence 和 semantic_notation，不得复用顶层字段，也不得省略。顶层字段是对所有事件的汇总，事件字段是对单个行为的刻画。"
                ),
                "driven_by": "social_relation",
                "fields": {
                    "social_relation": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{participants: [str], relation_type: [str], explicit_relation_statement: [str], evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "social_relation 对象仅在用户输入显式描述了 participants 中某角色的社会关系行为时生成",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{participants: [str], relation_type: [str], explicit_relation_statement: [str], evidence: [str], semantic_notation: str}]",
                                "description": "events 列表仅在用户输入显式描述了 participants 中某角色的社会关系行为时生成",
                                "items": {
                                    "experiencer": {
                                        "type": "array",
                                        "required": False,
                                        "description": "该关系事件中涉及的所有参与者，必须全部已在系统 participants 列表中定义。若提及未注册角色，则不得包含该关系事件"
                                    },
                                    "relation_type": {
                                        "type": "array",
                                        "required": False,
                                        "description": "从 explicit_relation_statement 中直接提取的关系类型关键词，必须为原文显式出现或可无歧义提取的词，禁止泛化"
                                    },
                                    "explicit_relation_statement": {
                                        "type": "array",
                                        "required": False,
                                        "description": "用户输入中直接陈述关系的完整短语或句子，必须逐字或近乎逐字引用"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须为当前社会关系事件提供直接支持的原文片段。仅包含用户输入中明确描述该社会关系事件的逐字语句，不得改写、总结、推理，也不得包含其他事件的内容"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须为当前社会关系事件生成一个高度提炼的语义标识符，仅反映该事件的核心社会关系行为模式。格式：snake_case，≤64字符，无数字，无专有名词，禁止跨事件归纳或抽象推理"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若输出 social_relation 对象，则此字段必须存在，且仅包含用户输入中直接支持时间判断的原文片段，逐字引用，不得改写、总结或推理"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若输出 social_relation 对象，则此字段必须存在，且必须为高度提炼的时间事件语义标识符（严格遵守 snake_case 格式，≤64 字符，无专有名词，无数字，仅描述显式或直接可推的语义模式）"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "仅当 social_relation 对象被输出时，≤100字，客观概括，不得引入新信息"
                            }
                        }
                    }
                }
            },

            # 其他的串行步骤
            # 推理层
            {
                "step": LLM_INFERENCE,
                "type": SERIAL,
                "index": 13,
                "label": "推理层：大模型基于感知数据合理推演",
                "role": "你是**心海系统**的超级推理专家",
                "sole_mission": "你的唯一任务是：基于注入的**已被验证的感知描述锚点**，进行最小必要、可追溯、可证伪的因果或动机推演。禁止从原始用户输入中直接推断；所有推理必须明确引用感知层输出作为依据。输出必须客观、克制、避免过度解读",
                "driven_by": "inference",
                "fields": {
                    "inference": {
                        "type": "object",
                        "required": True,
                        "format": "{events: [{experiencer: str, inference_type: str, anchor_points: [str], inferred_proposition: str, evidence: [str], semantic_notation: str}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "所有推理必须严格绑定到感知层输出的 semantic_notation。每个 events 条目独立完整，不得省略任何字段（若无法生成有效推理，则该条目应整体省略）。顶层 evidence 是所有 events.evidence 的去重并集；顶层 semantic_notation 和 summary 仅在 events 非空时生成。",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{inference_type: str, anchor_points: [str], inferred_proposition: str, evidence: [str], semantic_notation: str}]",
                                "description": "每个推理事件必须满足：\n1) experiencer 必须在 participants 列表中；\n2) inference_type 从预定义枚举中选择；\n3) anchor_points 必须引用感知层已生成的 semantic_notation；\n4) inferred_proposition 必须以‘可能’‘似乎’‘暗示’等弱化词开头，仅描述单一主体的单一动因或因果，≤100字；\n5) evidence 必须严格等于 anchor_points 所对应感知事件的原始 evidence 字段内容；\n6) semantic_notation 格式为：inference_{inference_type}_{key_concept}（snake_case，≤64字符，无数字/专有名词）。",
                                "items": {
                                    "experiencer": {
                                      "type": "string",
                                      "required": False,
                                      "description": "该推理所针对的主体，必须是 participants 中的 entity"
                                    },
                                    "inference_type": {
                                        "type": "string",
                                        "required": False,
                                        "enum": [
                                            "motivational",
                                            "causal",
                                            "intentional",
                                            "conflict_root",
                                            "expectation_violation",
                                            "pattern_recognition",
                                            "counterfactual"
                                        ],
                                        "description": (
                                            "- motivational: 行为背后可推断的驱动力（如需求、目标、恐惧）"
                                            "- causal: 某状态（情绪/行为）的直接触发因素"
                                            "- intentional: 未明说但可从行为模式中合理推断的目的"
                                            "- conflict_root: 多方行为或立场不可调和的核心分歧点"
                                            "- expectation_violation: 对显式或隐式预期（如承诺、惯例、声明）的偏离"
                                            "- pattern_recognition: 重复出现的行为序列或互动结构"
                                            "- counterfactual: 对“若当时不同”情境的假设性反思"
                                        )
                                    },
                                    "anchor_points": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须严格引用感知层输出中存在的 semantic_notation 标签，每个标签对应一个已被验证的感知事件。禁止组合、推导或引用未在感知层显式生成的标签"
                                    },
                                    "inferred_proposition": {
                                        "type": "string",
                                        "required": False,
                                        "description": "必须以“可能”“似乎”“暗示”等弱化词开头，仅描述单一 experiencer 的一个可观察动因、意图或因果关联（如情绪诱因、行为目的），不得引入未在感知层出现的新实体、关系或未经锚定的心理状态。结论必须能通过后续对话中的新 evidence 被证伪或修正。≤100 字。"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "必须严格等于 anchor_points 中每个 semantic_notation 在感知层对应的原始 evidence 字段内容。不得增删、改写、合并或跨事件引用。"
                                    },
                                    "semantic_notation": {
                                        "type": "string",
                                        "required": False,
                                        "description": "格式：inference_{inference_type}_{key_concept}，使用 snake_case，≤64 字符，无数字、无专有名词。key_concept 应取自 inferred_proposition 中的核心名词。禁止跨事件归纳。"
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若 events 非空，则此字段必须存在，且为所有 events[].evidence 的严格去重并集（顺序无关）。不得添加任何未在 events.evidence 中出现的文本。"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description":  "若 events 非空，则此字段必须存在。格式：inference_scene_{dominant_inference_type}_{core_theme}，使用 snake_case，≤64 字符。dominant_inference_type 取 events 中出现频次最高的类型；core_theme 应概括 events 中共现的核心动因或互动特征（如 control, withdrawal, obligation, inconsistency）。"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "若 events 非空，则此字段必须存在，≤100 字。仅客观复述 events 中的推理结论，使用去角色化语言（如“一方”“该主体”），不得引入新信息、评价、建议或情感色彩"
                            }
                        }
                    }
                }
            },

            # 深度分析
            {
                "step": LLM_DEEP_ANALYSIS,
                "type": SERIAL,
                "index": 14,
                "label": "深度分析层：剥离表象，基于人类底层心理动因进行去角色化分析",
                "role": "你是**心海系统**的超级人性底层架构师，要祛魅化，去角色化，去情感化。",
                "sole_mission": "你的唯一任务是：从注入的上下文描述中提取与字段模板完全匹配的底层动因描述。禁止推测、禁止填充默认值、禁止道德预设。输出必须是客观、可验证、去角色化的最小事实单元",
                "driven_by": "deep_analysis",
                "fields": {
                    "deep_analysis": {
                        "type": "object",
                        "required": False,
                        "format": "{events: [{core_driver: [str], power_asymmetry: {...}, resource_control: [str], survival_imperative: [str], social_enforcement_mechanism: [str], narrative_distortion: {...}}], evidence: [str], semantic_notation: str, summary: str}",
                        "description": "所有分析必须严格基于推理层（inference）的整体上下文结构进行判断，优先依据每个推理事件的 inferred_proposition（推理结论）作为动因识别的核心依据，辅以 evidence（原始文本支撑）、anchor_points（语义锚点）和 inference_type（推理类型）进行交叉验证。仅当推理层未显式覆盖某动因维度，但用户输入中存在与该推理事件 anchor_points 语义一致的字面明确陈述时，才允许引用用户输入作为补充依据。禁止脱离推理层结构、仅凭原始文本自由推导。",
                        "items": {
                            "events": {
                                "type": "array",
                                "required": False,
                                "format": "[{core_driver: [str], power_asymmetry: {control_axis: [str], dependency_ratio: float, threat_vector: [str], evidence: [str]}, resource_control: [str], survival_imperative: [str], social_enforcement_mechanism: [str], narrative_distortion: {self_justification: str, blame_shift: str, moral_licensing: str, evidence: [str]}}]",
                                "description": "每个条目必须严格对应推理层 events 中的一个推理事件。字段填充以该事件的 inferred_proposition 为首要判断依据，evidence 用于验证其字面基础，anchor_points 用于界定语义边界。若 inferred_proposition 已隐含某类底层动因（如‘通过强调牺牲影响决定’隐含 moral_licensing 或 blame_shift），且 evidence 中存在对应原句，则可生成相应字段；若推理层未覆盖但用户输入中有与 anchor_points 强相关的显式语句，可谨慎补充。",
                                "items": {
                                    "experiencer": {
                                      "type": "string",
                                      "required": False,
                                      "description": "仅当推理事件的 inferred_proposition 或 anchor_points 能无歧义指派主体时填写；否则省略。"
                                    },
                                    "core_driver": {
                                        "type": "array",
                                        "required": False,
                                        "items": {"type": "string"},
                                        "description": "仅当 inferred_proposition 明确表达根本诉求，且 evidence 中有对应原句时，无损提炼为底层动因。禁止将 inferred_proposition 抽象化或泛化"
                                    },
                                    "care_expression": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当 inferred_proposition 指出明确表达的关怀行为或意图，且 evidence 中明确提及相关描述或行为时提取使用；否则应省略。"
                                    },
                                    "separation_anxiety": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当 inferred_proposition 指出因分离而显式陈述的担忧、恐惧或回忆，且 evidence 中明确提及相关描述或行为时提取使用；否则应省略。"
                                    },
                                    "protective_intent": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当 inferred_proposition 指出为对方健康、安全或福祉采取行动的直接表述，且 evidence 中明确提及相关描述或行为时提取使用；否则应省略。"
                                    },
                                    "power_asymmetry": {
                                        "type": "object",
                                        "required": False,
                                        "format": "{control_axis: [str], dependency_ratio: float, threat_vector: [str], evidence: [str]}",
                                        "description": "仅当 inferred_proposition 描述了权力施压行为（如‘试图通过情感勒索控制决定’），且 evidence 中包含显式控制/威胁语句（如‘你就不能…’隐含条件交换），才生成该对象。dependency_ratio 仅在 evidence 出现量化依赖（如‘全靠你’）时输出。",
                                        "items": {
                                            "control_axis": {
                                                "type": "array",
                                                "required": False,
                                                "description": "基于 inferred_proposition 识别的控制维度（如情感绑定、义务施加、资源条件化），但必须有 evidence 中的字面支撑（如条件句、义务宣称、交换暗示），否则应省略。"
                                            },
                                            "dependency_ratio": {
                                                "type": "float",
                                                "required": False,
                                                "minimum": 0.00,
                                                "maximum": 1.00,
                                                "description": "仅当 evidence 中出现明确依赖程度表述时输出；否则彻底省略。"
                                            },
                                            "threat_vector": {
                                                "type": "array",
                                                "required": False,
                                                "description": "仅当 evidence 中直接陈述若不服从将导致的负面后果（如关系断裂、情感撤回、社会惩罚、资源剥夺），且 inferred_proposition 将其解释为施压手段时填写"
                                            },
                                            "evidence": {
                                                "type": "array",
                                                "required": False,
                                                "description": "必须为支撑 power_asymmetry 的原始文本片段，逐字来自推理层 evidence 或（在推理层未覆盖时）用户输入中与 anchor_points 一致的显式语句。"
                                            }
                                        }
                                    },
                                    "resource_control": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当 inferred_proposition 指出某方掌控具体、可操作的资源（如经济、法律、居住权），且 evidence 中明确提及相关描述或行为时提取使用；否则应省略。"
                                    },
                                    "survival_imperative": {
                                        "type": "array",
                                        "required": False,
                                        "description": "仅当 inferred_proposition 表明服从出于生存/安全需求，且 evidence 中关联性地字面表达；否则应省略"
                                    },
                                    "social_enforcement_mechanism": {
                                        "type": "array",
                                        "required": False,
                                        "description": "当 inferred_proposition 指出行为受外部群体规范或社会评价压力驱动，且 evidence 中出现对第三方评判的显式引用（如提及“他人看法”“规范期待”“声誉风险”等），才可填写。该机制体现为通过调用共享社会脚本（如义务、忠诚、感恩）来强化服从"
                                    },
                                    "narrative_distortion": {
                                        "type": "object",
                                        "required": False,
                                        "format": "{self_justification: str, blame_shift: str, moral_licensing: str, evidence: [str]}",
                                        "description": "仅当 inferred_proposition 识别出典型认知扭曲话术模式（如自我合理化、责任外推、道德资本兑换），且 evidence 中存在与该模式语义一致的显式陈述时，才生成该对象。每种子类型必须严格对应其定义：self_justification（为自身行为直接辩解）、blame_shift（将负面结果归因于对方）、moral_licensing（以过往道德行为为当前要求或行为开脱）",
                                        "items": {
                                            "self_justification": {
                                                "type": "string",
                                                "required": False,
                                                "description": "仅当 evidence 中出现为自身行为提供直接正当性理由的陈述（如声称行为出于对方利益、必要性或善意），且 inferred_proposition 将其解释为自我合理化"
                                            },
                                            "blame_shift": {
                                                "type": "string",
                                                "required": False,
                                                "description": "仅当 evidence 中有明确将负面状态、情绪或后果归咎于对方的语句（如暗示“因你我才如此”“都是你导致…”），且 inferred_proposition 将其解释为责任转嫁"
                                            },
                                            "moral_licensing": {
                                                "type": "string",
                                                "required": False,
                                                "description": "仅当 evidence 中出现以过往道德付出、牺牲或善意行为作为当前要求、控制或豁免理由的陈述（如“我做了X，所以你必须Y”），且 inferred_proposition 支持该解读"
                                            },
                                            "evidence": {
                                                "type": "array",
                                                "required": False,
                                                "description": "必须为支撑 narrative_distortion 的原始文本片段，逐字来自推理层 evidence 或（在推理层未覆盖时）用户输入中与 anchor_points 一致的显式语句。"
                                            }
                                        }
                                    }
                                }
                            },
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "若 events 非空，则此字段必须存在，必须为所有 events 中引用的 evidence 字段的严格去重并集，全部内容必须可追溯至推理层 evidence 或经 anchor_points 验证的用户输入显式语句。"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "若 events 非空，则此字段必须存在，格式：deep_analysis_{primary_theme}，snake_case，≤64字符；primary_theme 必须基于推理层 inferred_proposition 中已识别的核心动因"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "若 events 非空，则此字段必须存在，≤100字；仅复述推理层中已通过 inferred_proposition 和 evidence 验证的显式动因"
                            }
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
                "role": "你是**心海系统**的超级感性与理性并重的人生导师。",
                "sole_mission": "你的唯一任务是：仅当深度分析输出非空时，基于其**显式陈述的底层动因与结构约束**，提出最小可行、可追溯、无道德预设的行动建议。所有建议必须：\n- 引用深度分析层的具体字段作为依据\n- 仅使用用户已提及的资源或能力\n- 避免理想化、激进或脱离文化语境的方案。若深度分析部分字段缺失，可基于已有事件与用户原文进行最小必要推演，但须确保 semantic_notation 与 summary 始终可生成",
                "driven_by": "rational_advice",
                "fields": {
                    "rational_advice": {
                        "type": "object",
                        "required": False,
                        "format": "{evidence: [str], semantic_notation: str, summary: str, safety_first_intervention: [str], systemic_leverage_point: [str], incremental_strategy: [str], stakeholder_tradeoffs: {...}, long_term_exit_path: [str], cultural_adaptation_needed: [str]}",
                        "description": "仅当 deep_analysis.events 非空时，必须生成本对象（不得返回 null 或 {}）。其中 semantic_notation 与 summary 为逻辑必填字段：semantic_notation 应基于本层建议的核心策略命名；可依次从 deep_analysis.events → deep_analysis.summary → 用户原始输入中兜底生成。",
                        "items": {
                            "evidence": {
                                "type": "array",
                                "required": False,
                                "description": "引用所依据的 deep_analysis 对象的 semantic_notation，用于可追溯性"
                            },
                            "semantic_notation": {
                                "type": "string",
                                "required": False,
                                "description": "整体建议方案的标准化语义标识，格式：rational_advice_{primary_strategy}（snake_case，≤64字符）。primary_strategy 应概括本建议所针对的核心互动矛盾或杠杆点，优先基于 deep_analysis.events 中的显式动因（如 core_driver）提炼；若无，则从用户输入中抽象得出"
                            },
                            "summary": {
                                "type": "string",
                                "required": False,
                                "description": "≤100字；仅复述基于 deep_analysis 显式内容推导出的最小可行路径"
                            },
                            "safety_first_intervention": {
                                "type": "array",
                                "required": False,
                                "description": "优先确保低位者心理与关系安全的最小干预措施。每条必须标注可显示支持的依据字段路径（如 deep_analysis.events[0].protective_intent）。措施应聚焦于用户已具备的即时能力或环境（如沉默、回避、记录、短暂独处），避免引入外部依赖。"
                            },
                            "systemic_leverage_point": {
                                "type": "array",
                                "required": False,
                                "description": "可撬动当前权力结构的关键支点，必须严格对应 deep_analysis.events 中显式存在的 control_axis（控制轴）或 resource_control（资源控制点）。不得推测未提及的权力机制"
                            },
                            "incremental_strategy": {
                                "type": "array",
                                "required": False,
                                "description": "提供分阶段、低风险、可执行的行动策略，所有步骤必须由低位者（或受保护者）独立发起并完成，无需高位者知情、同意或回应。每条策略应包含：(1) 一个具体、可观测的行为；(2) 明确的执行时机或频率；(3) 所依赖的、用户已在输入中提及的资源；(4) 一个可识别的风险信号及应对方式。策略应将抽象的情感或关系需求，转化为基于现有条件的微小日常实践，避免任何需要外部配合或理想化假设的行动。"
                            },
                            "stakeholder_tradeoffs": {
                                "type": "object",
                                "required": False,
                                "format": "{victim_cost: [str], oppressor_loss: [str], system_stability: [str], evidence: [str]}",
                                "description": "各方代价评估，使用中性、非评判性语言，必须基于 deep_analysis 显式内容",
                                "items": {
                                    "victim_cost": {
                                        "type": "array",
                                        "required": False,
                                        "description": "低位者可能承担的社会、情感或生存风险，需对应 deep_analysis.events 中的 core_driver 或 survival_imperative。例如：“若减少某类服从行为，可能被强化负面身份标签（依据：deep_analysis.events[0].social_enforcement_mechanism）”。"
                                    },
                                    "oppressor_loss": {
                                        "type": "array",
                                        "required": False,
                                        "description": "高位者可能失去的控制手段、资源或象征性权力，需对应 deep_analysis.events 中的 control_axis 或 resource_control。例如：“若接受替代性互动方式，将削弱原有控制机制的即时效力（依据：deep_analysis.events[0].power_asymmetry.control_axis）”。"
                                    },
                                    "system_stability": {
                                        "type": "array",
                                        "required": False,
                                        "description": "对当前关系系统短期稳定性的潜在冲击，需对应 deep_analysis.events 中的 social_enforcement_mechanism。例如：“若未维持某种被期待的行为模式，可能引发外部评价压力，动摇系统内某方的社会合法性（依据：deep_analysis.events[0].social_enforcement_mechanism）”。"
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "required": False,
                                        "description": "支撑代价评估的 deep_analysis 字段路径，如 ['deep_analysis.events[0].power_asymmetry.threat_vector', 'deep_analysis.events[0].social_enforcement_mechanism']"
                                    }
                                }
                            },
                            "long_term_exit_path": {
                                "type": "array",
                                "required": False,
                                "description": "可持续脱离当前情感-社会绑定结构的现实路径，必须基于用户输入中隐含的长期可能性（如对某空间的向往、对独立生活的提及、对某资源的掌控意愿），但不得虚构用户未提及的能力、关系或资源。路径应聚焦于逐步积累可验证的自主性证据（如经济记录、社交边界、物理空间使用权），以降低未来脱离时的系统性反制风险。"
                            },
                            "cultural_adaptation_needed": {
                                "type": "array",
                                "required": False,
                                "description": "需调整的文化认知或可寻求的社会支持，必须对应 deep_analysis.events 中明确提及的社会规范或评价机制。例如：将某行为重新定义为符合本地可接受的规范形式，以降低 social_shaming 风险；或识别用户输入中已存在的潜在支持者（如亲属、同事、邻居），作为缓冲外部压力的见证节点（依据：deep_analysis.events[0].social_enforcement_mechanism）"
                            }
                        }
                    }
                }
            }
        ]
    },
    CATEGORY_SUGGESTION: {
        "common_suggestion": """你是一位资深临床心理分析师，习惯在深夜整理个案笔记。你的文字冷静、克制，但字里行间透出对人性复杂性的深刻理解。你不下判断，只呈现机制；不讲故事，只揭示结构。
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
            现在，开始陈述："""
    }
}
