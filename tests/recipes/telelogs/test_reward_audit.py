import json

from scripts.telelogs.audit_reward_alignment import audit_file, audit_record


def test_audit_flags_reward_quality_conflict_update():
    record = {
        "step": 1,
        "case_id": "c1",
        "rewards": [1.0, 0.0],
        "correctness_scores": [0.0, 1.0],
        "efficiency_costs": [2, 2],
        "learning_route": "audit_reward_quality_conflict",
        "optimizer_updated": True,
    }
    result = audit_record(record)
    assert result["recomputed_route"] == "audit_reward_quality_conflict"
    assert "unsafe_optimizer_update" in result["issues"]


def test_audit_accepts_aligned_rl_update():
    record = {
        "step": 2,
        "case_id": "c2",
        "rewards": [0.0, 1.0],
        "correctness_scores": [0.0, 1.0],
        "efficiency_costs": [3, 3],
        "learning_route": "rl_ready",
        "optimizer_updated": True,
    }
    result = audit_record(record)
    assert result["recomputed_route"] == "rl_ready"
    assert result["issues"] == []


def test_audit_file_reports_route_mismatch(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps(
            {
                "step": 0,
                "case_id": "c3",
                "rewards": [0.0, 1.0],
                "correctness_scores": [0.0, 1.0],
                "efficiency_costs": [1, 1],
                "learning_route": "teacher_or_sft_repair",
                "optimizer_updated": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = audit_file(path)
    assert report["records"] == 1
    assert report["issue_counts"]["route_mismatch"] == 1
    assert report["issue_counts"]["missed_optimizer_update"] == 1
