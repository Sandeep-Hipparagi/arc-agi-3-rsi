from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class TwoStageWorldModel:
    """
    L0 World Model for the active game instance.
    Strictly separates Unverified Hypotheses from Proven Mechanics & Fatal Transitions.
    """
    unverified_hypotheses: Dict[str, str] = field(default_factory=dict)
    proven_mechanics: Dict[str, str] = field(default_factory=dict)
    fatal_transitions: List[str] = field(default_factory=list)
    level_notes: Dict[int, str] = field(default_factory=dict)

    def update(
        self,
        op: str,
        key: str,
        description: str
    ) -> str:
        """
        Operations:
        - 'add_hypothesis': Add or update an unverified hypothesis
        - 'promote_to_proven': Move/record a rule into proven_mechanics (removes from hypotheses)
        - 'refute_hypothesis': Delete a falsified hypothesis
        - 'add_fatal': Record a fatal state/action rule ('never make the same mistake twice')
        """
        if op == "add_hypothesis":
            self.unverified_hypotheses[key] = description
            return f"Added/updated hypothesis [{key}]."
        elif op == "promote_to_proven":
            self.unverified_hypotheses.pop(key, None)
            self.proven_mechanics[key] = description
            return f"Promoted [{key}] to Proven Mechanics."
        elif op == "refute_hypothesis":
            self.unverified_hypotheses.pop(key, None)
            return f"Refuted and removed hypothesis [{key}]."
        elif op == "add_fatal":
            rule = f"[{key}]: {description}"
            if rule not in self.fatal_transitions:
                self.fatal_transitions.append(rule)
            return f"Recorded fatal transition rule: {rule}"
        return f"Unknown operation: {op}"

    def render_markdown(self) -> str:
        lines = ["## L0 Active Game World Model"]
        
        lines.append("\n### 1. Banned / Fatal Transitions (NEVER VIOLATE)")
        if self.fatal_transitions:
            for item in self.fatal_transitions:
                lines.append(f"- ⚠️ {item}")
        else:
            lines.append("- None recorded yet.")

        lines.append("\n### 2. Proven Mechanics (Verified Causal Laws)")
        if self.proven_mechanics:
            for k, v in self.proven_mechanics.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append("- None verified yet.")

        lines.append("\n### 3. Unverified Hypotheses (Requires Active Probing)")
        if self.unverified_hypotheses:
            for k, v in self.unverified_hypotheses.items():
                lines.append(f"- ? **{k}**: {v}")
        else:
            lines.append("- None active.")

        return "\n".join(lines)
