import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class SkillNode:
    node_id: str
    parent_id: Optional[str]
    generation: int
    skill_path: str
    learner_policy_path: str
    mutation_rationale: str
    game_scores: Dict[str, float] = field(default_factory=dict)
    level_scores_by_game: Dict[str, List[float]] = field(default_factory=dict)
    mean_batch_score: float = 0.0
    tiebreaker_score: float = 0.0
    total_actions_across_batch: int = 0
    passed_regression_gate: bool = True
    rejection_reason: Optional[str] = None
    run_dirs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillNode":
        return cls(**data)


class EvolutionTree:
    """
    Manages the lineage tree of evaluated skill nodes, enforces the strict
    regression gate, and persists search state to `lineage_tree.json`.
    """
    def __init__(self, experiment_dir: Path, experiment_id: str):
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.tree_file = self.experiment_dir / "lineage_tree.json"
        self.nodes: Dict[str, SkillNode] = {}
        self.root_node_id: Optional[str] = None

        if self.tree_file.exists():
            self._load()

    @classmethod
    def load_from_checkpoint(cls, experiment_dir: Path) -> "EvolutionTree":
        """
        Loads an existing evolution tree from experiment_dir / 'lineage_tree.json'.
        """
        tree_file = experiment_dir / "lineage_tree.json"
        if not tree_file.exists():
            raise FileNotFoundError(f"No lineage_tree.json found in {experiment_dir}")
        data = json.loads(tree_file.read_text(encoding="utf-8"))
        exp_id = data.get("experiment_id", experiment_dir.name)
        tree = cls(experiment_dir=experiment_dir, experiment_id=exp_id)
        return tree

    def get_max_generation(self) -> int:
        """
        Returns the highest generation number present in the tree, or -1 if empty.
        """
        if not self.nodes:
            return -1
        return max(n.generation for n in self.nodes.values())

    def add_node(self, node: SkillNode) -> None:
        self.nodes[node.node_id] = node
        if node.parent_id is None and self.root_node_id is None:
            self.root_node_id = node.node_id
        self.save()

    def check_regression_gate(self, parent: SkillNode, candidate: SkillNode) -> Tuple[bool, str]:
        """
        Enforces two strict conditions:
        1. Non-Regression on Mastered Levels:
           For every game and level where Parent achieved score >= 1.0,
           Candidate must also achieve score >= 1.0.
        2. Strict Improvement:
           Candidate must have a higher mean_batch_score, OR equal mean_batch_score
           with a higher tiebreaker_score and fewer total actions.
        """
        # Condition 1: Check regressions on mastered levels
        for game_id, p_levels in parent.level_scores_by_game.items():
            c_levels = candidate.level_scores_by_game.get(game_id, [])
            for lvl_idx, p_score in enumerate(p_levels):
                if p_score >= 1.0:
                    c_score = c_levels[lvl_idx] if lvl_idx < len(c_levels) else 0.0
                    if c_score < 1.0:
                        reason = (
                            f"Regressed on mastered level: {game_id} Level {lvl_idx+1} "
                            f"(parent score={p_score:.2f}, candidate score={c_score:.2f})"
                        )
                        return False, reason

        # Condition 2: Strict improvement
        if candidate.mean_batch_score > parent.mean_batch_score:
            return True, f"Improved mean batch score: {parent.mean_batch_score:.4f} -> {candidate.mean_batch_score:.4f}"

        if candidate.mean_batch_score == parent.mean_batch_score:
            if candidate.tiebreaker_score > parent.tiebreaker_score:
                return True, (
                    f"Tiebreaker improvement at score {candidate.mean_batch_score:.4f}: "
                    f"tiebreaker {parent.tiebreaker_score:.2f} -> {candidate.tiebreaker_score:.2f}"
                )
            elif (
                candidate.tiebreaker_score == parent.tiebreaker_score
                and candidate.total_actions_across_batch < parent.total_actions_across_batch
            ):
                return True, (
                    f"Efficiency improvement: actions {parent.total_actions_across_batch} -> "
                    f"{candidate.total_actions_across_batch}"
                )
            else:
                return False, (
                    f"No improvement: equal score ({candidate.mean_batch_score:.4f}) without better tiebreaker "
                    f"({candidate.tiebreaker_score:.2f} <= {parent.tiebreaker_score:.2f}) or action reduction"
                )

        return False, (
            f"Lower batch score: candidate ({candidate.mean_batch_score:.4f}) < parent ({parent.mean_batch_score:.4f})"
        )

    def select_best_frontier_node(self) -> SkillNode:
        """
        Returns the highest-scoring node that passed the regression gate.
        Falls back to the root node if no candidates passed.
        """
        valid_nodes = [n for n in self.nodes.values() if n.passed_regression_gate]
        if not valid_nodes:
            return self.nodes[self.root_node_id] if self.root_node_id else list(self.nodes.values())[0]

        # Sort primarily by mean_batch_score, then tiebreaker_score, then negative total_actions
        sorted_nodes = sorted(
            valid_nodes,
            key=lambda n: (n.mean_batch_score, n.tiebreaker_score, -n.total_actions_across_batch),
            reverse=True,
        )
        return sorted_nodes[0]

    def render_lineage_report(self) -> str:
        lines = [
            f"# ARC-AGI-3 Evolutionary Lineage Report",
            f"**Experiment ID**: `{self.experiment_id}`",
            f"**Total Evaluated Nodes**: {len(self.nodes)}",
            "",
            "| Node ID | Gen | Parent | Batch Score | Tiebreaker | Actions | Gate Status | Rationale |",
            "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        # Group by generation
        generations = sorted(set(n.generation for n in self.nodes.values()))
        for gen in generations:
            gen_nodes = [n for n in self.nodes.values() if n.generation == gen]
            for n in gen_nodes:
                status_icon = "[PASS] ACCEPTED" if n.passed_regression_gate else f"[FAIL] REJECTED ({n.rejection_reason or 'Failed gate'})"
                if n.parent_id is None:
                    status_icon = "[ROOT]"
                lines.append(
                    f"| `{n.node_id}` | {n.generation} | `{n.parent_id or '-'}` | "
                    f"{n.mean_batch_score:.4f} | {n.tiebreaker_score:.2f} | {n.total_actions_across_batch} | "
                    f"{status_icon} | {n.mutation_rationale[:60]}... |"
                )

        best_node = self.select_best_frontier_node()
        lines.extend([
            "",
            f"### Winning Frontier Skill: `{best_node.node_id}`",
            f"- **Mean Batch Score**: {best_node.mean_batch_score:.4f}",
            f"- **Tiebreaker**: {best_node.tiebreaker_score:.2f}",
            f"- **Total Actions**: {best_node.total_actions_across_batch}",
            f"- **Skill Path**: `{best_node.skill_path}`",
        ])
        return "\n".join(lines)

    def save(self) -> None:
        data = {
            "experiment_id": self.experiment_id,
            "root_node_id": self.root_node_id,
            "best_node_id": self.select_best_frontier_node().node_id if self.nodes else None,
            "timestamp": time.time(),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }
        self.tree_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        try:
            data = json.loads(self.tree_file.read_text(encoding="utf-8"))
            self.root_node_id = data.get("root_node_id")
            self.nodes = {nid: SkillNode.from_dict(nd) for nid, nd in data.get("nodes", {}).items()}
        except Exception:
            self.nodes = {}
            self.root_node_id = None
