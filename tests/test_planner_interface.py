from app.concierge.planner_interface import PlannerInterface


def test_plan_preview_structure(tmp_path):
    (tmp_path / "BankFAQs.csv").write_text("question,answer\nQ?,A?")
    planner = PlannerInterface(vertical="retail", data_dir=tmp_path)
    plan = planner.generate_plan_preview(use_llm=False)

    assert plan["type"] == "factory_plan_preview"
    assert "agents" in plan
    assert isinstance(plan["agents"], list)
    assert any(a["id"] == "guardrails" for a in plan["agents"])
    assert "actions" in plan
