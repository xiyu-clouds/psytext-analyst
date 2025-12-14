from src.state_of_mind.stages.perception.constants import CATEGORY_RAW, LLM_PARTICIPANTS_EXTRACTION
from src.state_of_mind.stages.perception.data_validator import DataValidator


def test_participants_extraction_with_repair():
    validator = DataValidator(auto_repair=True)

    # 模拟 LLM 输出：部分字段不是 list（但规则要求是 list），触发自动修复
    input_data = {
        "__meta_id": "test_run_001",
        "participants": [
            {
                "entity": "张三",
                "social_role": "父亲",
                "cultural_identity": "汉族",  # ❌ 应为 list，但给了 str
                "physical_traits": ["高个子"],  # ✅ 正确
                "carried_objects": "黑色公文包",  # ❌ 应为 list，但给了 str
                "personality_traits": None,  # ❌ 必填？不，非必填但为空 → 被清理
            },
            {
                "entity": "李四",
                "occupation": "软件工程师",
                "cultural_identity": ["汉族", "北京人"],  # ✅ 正确
                "appearance": "戴眼镜、穿格子衬衫",  # ❌ 应为 list
            }
        ]
    }

    result = validator.validate(input_data, CATEGORY_RAW, LLM_PARTICIPANTS_EXTRACTION)

    print("✅ Validation passed:", result["is_valid"])
    print("\n📋 Errors:")
    for e in result["errors"]:
        print("  -", e)

    cleaned = result["cleaned_data"]
    print("\n🔍 Cleaned Data (participants[0]):")
    p0 = cleaned["participants"][0]
    print(f"  entity: {p0['entity']}")
    print(f"  cultural_identity: {p0.get('cultural_identity')} (type: {type(p0.get('cultural_identity'))})")
    print(f"  carried_objects: {p0.get('carried_objects')} (type: {type(p0.get('carried_objects'))})")

    print("\n🔍 Cleaned Data (participants[1]):")
    p1 = cleaned["participants"][1]
    print(f"  appearance: {p1.get('appearance')} (type: {type(p1.get('appearance'))})")

    # === 关键断言 ===
    assert isinstance(cleaned["participants"], list), "Top-level 'participants' must remain a LIST"
    assert len(cleaned["participants"]) == 2, "Participant count must be preserved"

    p0 = cleaned["participants"][0]
    assert p0["entity"] == "张三"
    assert p0["cultural_identity"] == ["汉族"], "Should be auto-repaired to list"
    assert p0["carried_objects"] == ["黑色公文包"], "Should be auto-repaired to list"
    assert "personality_traits" not in p0, "Empty optional field should be removed"

    p1 = cleaned["participants"][1]
    assert p1["appearance"] == ["戴眼镜、穿格子衬衫"], "Should be auto-repaired to list"

    # 检查错误信息（因为 repair 成功，不应有类型错误）
    # 注意：cultural_identity 等是非必填，所以即使原值是 str，repair 后通过，无 error
    assert len(result["errors"]) == 0, "All fields should pass after auto-repair"

    print("\n🎉 测试通过：结构保持完整，自动修复生效，无错误！")


if __name__ == "__main__":
    test_participants_extraction_with_repair()
