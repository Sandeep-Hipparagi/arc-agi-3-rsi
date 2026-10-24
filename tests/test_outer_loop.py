import json
from pathlib import Path
import pytest

from src.trace_digest import TraceDigester
from src.evolution_tree import EvolutionTree, SkillNode
from src.learner_agent import LearnerAgent, SkillCandidate
from evolve import run_evolution


def test_trace_digester_noop_and_cycle_detection(tmp_path: Path):
    """
    Create a synthetic run with repeated state_hash values and grid_changed=False events,
    run TraceDigester.digest_run(), and assert accurate counts for noop_steps,
    cyclic_revisits, and repl_error_count, plus verify Markdown digest length is < 10,000 chars.
    """
    summary_data = {
        "run_id": "test_run_01",
        "game_id": "test_game",
        "final_status": "NOT_FINISHED",
        "levels_completed": 0,
        "total_env_steps": 6,
        "final_world_model": {
            "proven_mechanics": {"p1": "Wall is solid"},
            "unverified_hypotheses": {"h1": "Red might be dangerous"},
            "fatal_transitions": ["Never hit red"],
        },
        "official_scorecard": {
            "score": 0.0,
            "environments": [{
                "id": "test_game",
                "runs": [{
                    "level_scores": [0.0],
                    "level_actions": [6],
                    "level_baseline_actions": [20],
                    "resets": 0,
                }],
            }],
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary_data), encoding="utf-8")

    trajectory_events = [
        # Event 1: REPL error
        {
            "event": "repl_exec",
            "levels_completed": 0,
            "state_status": "NOT_FINISHED",
            "payload": {
                "code": "print(foo)",
                "output": "Traceback (most recent call last):\nNameError: name 'foo' is not defined",
            },
        },
        # Event 2: Valid move to hash_a
        {
            "event": "env_step",
            "levels_completed": 0,
            "state_status": "NOT_FINISHED",
            "payload": {
                "action": "ACTION1",
                "thought": "Move UP",
                "state_hash": "hash_a",
                "prev_state_hash": "hash_0",
                "grid_changed": True,
            },
        },
        # Event 3: No-op bump (grid_changed=False, state_hash=hash_a -> cyclic revisit!)
        {
            "event": "env_step",
            "levels_completed": 0,
            "state_status": "NOT_FINISHED",
            "payload": {
                "action": "ACTION1",
                "thought": "Move UP again into wall",
                "state_hash": "hash_a",
                "prev_state_hash": "hash_a",
                "grid_changed": False,
            },
        },
        # Event 4: No-op bump 2
        {
            "event": "env_step",
            "levels_completed": 0,
            "state_status": "NOT_FINISHED",
            "payload": {
                "action": "ACTION1",
                "thought": "Move UP again into wall",
                "state_hash": "hash_a",
                "prev_state_hash": "hash_a",
                "grid_changed": False,
            },
        },
        # Event 5: No-op bump 3 (creates waste cluster >= 3)
        {
            "event": "env_step",
            "levels_completed": 0,
            "state_status": "NOT_FINISHED",
            "payload": {
                "action": "ACTION1",
                "thought": "Move UP again into wall",
                "state_hash": "hash_a",
                "prev_state_hash": "hash_a",
                "grid_changed": False,
            },
        },
        # Event 6: Valid move to hash_b
        {
            "event": "env_step",
            "levels_completed": 0,
            "state_status": "NOT_FINISHED",
            "payload": {
                "action": "ACTION2",
                "thought": "Move DOWN to escape",
                "state_hash": "hash_b",
                "prev_state_hash": "hash_a",
                "grid_changed": True,
            },
        },
    ]

    with (tmp_path / "trajectory.jsonl").open("w", encoding="utf-8") as f:
        for ev in trajectory_events:
            f.write(json.dumps(ev) + "\n")

    digest = TraceDigester.digest_run(tmp_path)

    # Assert accurate hygiene counts
    assert digest["hygiene"]["noop_steps"] == 3
    assert digest["hygiene"]["cyclic_revisits"] == 3
    assert digest["hygiene"]["repl_error_count"] == 1
    assert "NameError: name 'foo' is not defined" in digest["hygiene"]["unique_repl_errors"][0]
    assert len(digest["waste_clusters"]) >= 1
    assert digest["waste_clusters"][0]["count"] == 3

    # Render Markdown report and assert token efficiency
    md = TraceDigester.render_digest_markdown(digest)
    assert len(md) < 10000, f"Digest markdown exceeds 10,000 characters: {len(md)}"
    assert "Failure Mode 1 (Wall Bumping)" in md
    assert "Diagnostic Digest" in md


def test_regression_gate_blocks_mastered_level_drop(tmp_path: Path):
    """
    Construct a parent SkillNode with level_scores_by_game={"ls20": [1.0, 0.5]} and a
    candidate SkillNode with level_scores_by_game={"ls20": [0.8, 0.9]} (higher average,
    but regressed on mastered Level 1). Assert check_regression_gate() returns False.
    """
    tree = EvolutionTree(experiment_dir=tmp_path, experiment_id="test_reg")

    parent = SkillNode(
        node_id="parent_root",
        parent_id=None,
        generation=0,
        skill_path="skills/SKILL.md",
        learner_policy_path="skills/L2_learner_policy.md",
        mutation_rationale="Parent Root",
        game_scores={"ls20": 0.75},
        level_scores_by_game={"ls20": [1.0, 0.5]},
        mean_batch_score=0.75,
        tiebreaker_score=10.0,
        total_actions_across_batch=30,
        passed_regression_gate=True,
    )
    tree.add_node(parent)

    # Candidate with higher mean (0.85 > 0.75), but regressed on mastered Level 1 (0.8 < 1.0)
    candidate_regressed = SkillNode(
        node_id="cand_regressed",
        parent_id="parent_root",
        generation=1,
        skill_path="skills/SKILL.md",
        learner_policy_path="skills/L2_learner_policy.md",
        mutation_rationale="Candidate Regressed",
        game_scores={"ls20": 0.85},
        level_scores_by_game={"ls20": [0.8, 0.9]},
        mean_batch_score=0.85,
        tiebreaker_score=15.0,
        total_actions_across_batch=25,
    )

    passed, reason = tree.check_regression_gate(parent, candidate_regressed)
    assert passed is False, "Regression gate should have blocked the candidate!"
    assert "Regressed on mastered level" in reason
    assert "Level 1" in reason

    # Candidate with strict improvement (retains 1.0 on Level 1, improves Level 2 to 0.9)
    candidate_valid = SkillNode(
        node_id="cand_valid",
        parent_id="parent_root",
        generation=1,
        skill_path="skills/SKILL.md",
        learner_policy_path="skills/L2_learner_policy.md",
        mutation_rationale="Candidate Valid Improvement",
        game_scores={"ls20": 0.95},
        level_scores_by_game={"ls20": [1.0, 0.9]},
        mean_batch_score=0.95,
        tiebreaker_score=15.0,
        total_actions_across_batch=22,
    )

    passed_valid, reason_valid = tree.check_regression_gate(parent, candidate_valid)
    assert passed_valid is True, f"Valid candidate should pass gate: {reason_valid}"
    assert "Improved mean batch score" in reason_valid


def test_learner_agent_guardrails_and_dry_run():
    """
    Run LearnerAgent in dry_run=True mode to generate 2 offspring, and verify
    that the validation guardrail rejects any candidate containing hardcoded game IDs
    or missing required section headers.
    """
    learner = LearnerAgent(dry_run=True)
    parent_md = Path("skills/SKILL.md").read_text(encoding="utf-8")
    policy_md = Path("skills/L2_learner_policy.md").read_text(encoding="utf-8")

    offspring = learner.propose_offspring(
        parent_skill_markdown=parent_md,
        learner_policy_markdown=policy_md,
        batch_digests_markdown="Mock digest",
        num_offspring=2,
        generation=1,
    )

    assert len(offspring) == 2
    for child in offspring:
        is_valid, msg = LearnerAgent.validate_candidate(child)
        assert is_valid is True, f"Generated offspring failed validation: {msg}"
        assert child.variant_id.startswith("gen1_")
        assert len(child.mutation_rationale) > 10

    # Test rejection of hardcoded game ID
    bad_cand_game = SkillCandidate(
        variant_id="bad_cand_game",
        mutation_rationale="Hardcoding game solution",
        mutated_skill_markdown="# Domain Skill\n## OBSERVE\n## GROW\n## PLAN\n## EXECUTE\nFor game ls20 move UP 5 times.",
    )
    is_valid, msg = LearnerAgent.validate_candidate(bad_cand_game)
    assert is_valid is False
    assert "Domain-generalization violation" in msg
    assert "ls20" in msg

    # Test rejection of missing operating sections
    bad_cand_incomplete = SkillCandidate(
        variant_id="bad_cand_incomplete",
        mutation_rationale="Missing sections",
        mutated_skill_markdown="# Partial Skill\n## OBSERVE\nOnly observation here.",
    )
    is_valid2, msg2 = LearnerAgent.validate_candidate(bad_cand_incomplete)
    assert is_valid2 is False
    assert "Missing required core operating section" in msg2


def test_end_to_end_evolution_pipeline_mock(tmp_path: Path):
    """
    Run a 2-generation, 2-branch evolutionary sweep via evolve.py with
    --mock --dry-run --games mock_1,mock_2 --generations 2 --branch-factor 2 --max-steps 15,
    and assert that lineage_tree.json and best_SKILL.md are created and valid.
    """
    tree = run_evolution(
        games=["mock_1", "mock_2"],
        generations=2,
        branch_factor=2,
        max_steps=15,
        seed_skill_path=Path("skills/SKILL.md"),
        seed_policy_path=Path("skills/L2_learner_policy.md"),
        is_mock=True,
        is_dry_run=True,
        exp_dir=tmp_path / "evolution_runs",
    )

    assert tree.tree_file.exists(), "lineage_tree.json was not created!"
    # Gen 0 has 1 node, Gen 1 has 2 offspring, Gen 2 has 2 offspring = at least 5 nodes
    assert len(tree.nodes) >= 5, f"Expected >= 5 nodes, found {len(tree.nodes)}"

    best_skill = tmp_path / "evolution_runs" / tree.experiment_id / "best_SKILL.md"
    assert best_skill.exists(), "best_SKILL.md was not created!"
    best_text = best_skill.read_text(encoding="utf-8")
    assert len(best_text) > 100
    assert "OBSERVE" in best_text

    report_file = tmp_path / "evolution_runs" / tree.experiment_id / "lineage_report.md"
    assert report_file.exists(), "lineage_report.md was not created!"
    report_text = report_file.read_text(encoding="utf-8")
    assert "Evolutionary Lineage Report" in report_text
    assert "Winning Frontier Skill" in report_text
