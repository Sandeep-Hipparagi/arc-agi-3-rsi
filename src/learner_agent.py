import os
import re
import json
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI


@dataclass
class SkillCandidate:
    variant_id: str
    mutation_rationale: str
    mutated_skill_markdown: str
    mutated_learner_policy_markdown: Optional[str] = None


FORBIDDEN_GAME_PATTERNS = ["ls20", "ft09", "vc33", "mock_maze", "mock_1", "mock_2"]
REQUIRED_SECTIONS = ["OBSERVE", "GROW", "PLAN", "EXECUTE"]


class LearnerAgent:
    """
    Outer-Loop Learner Agent powered by native `openai.OpenAI` SDK.
    Inspects multi-game diagnostic digests, formulates structural hypotheses,
    and proposes K diverse mutated skill offspring.
    """
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.model = model
        self.dry_run = dry_run
        if not dry_run:
            resolved_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
            self.client: Optional[OpenAI] = OpenAI(base_url=base_url, api_key=resolved_key)
        else:
            self.client = None

    def _call_llm_with_retry(self, call_fn, max_retries: int = 4, **kwargs) -> Any:
        """
        Executes an LLM API call with exponential backoff and jitter for transient errors (429, 5xx, timeouts).
        """
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                return call_fn(**kwargs)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as e:
                if attempt == max_retries:
                    raise
                jitter = random.uniform(0.05, 0.2)
                time.sleep(backoff + jitter)
                backoff *= 2.0
            except openai.APIError as e:
                status = getattr(e, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    jitter = random.uniform(0.05, 0.2)
                    time.sleep(backoff + jitter)
                    backoff *= 2.0
                else:
                    raise

    @staticmethod
    def validate_candidate(candidate: SkillCandidate) -> Tuple[bool, str]:
        """
        Validates the candidate against strict domain-generalization and structural rules.
        """
        if not candidate.mutated_skill_markdown or not candidate.mutated_skill_markdown.strip():
            return False, "Candidate skill markdown is empty."

        md_upper = candidate.mutated_skill_markdown.upper()
        for sec in REQUIRED_SECTIONS:
            if sec not in md_upper:
                return False, f"Missing required core operating section: '{sec}'"

        md_lower = candidate.mutated_skill_markdown.lower()
        for forbidden in FORBIDDEN_GAME_PATTERNS:
            # Match whole-word or token occurrence of forbidden game IDs
            if re.search(r"\b" + re.escape(forbidden) + r"\b", md_lower):
                return False, f"Domain-generalization violation: hardcoded game reference '{forbidden}' detected."

        return True, "Valid"

    def _generate_dry_run_offspring(
        self,
        parent_skill_markdown: str,
        num_offspring: int,
        generation: int = 1
    ) -> List[SkillCandidate]:
        """
        Generates deterministic, syntactically verified child candidates for offline testing.
        """
        candidates: List[SkillCandidate] = []

        # Offspring 1: Spatial 3x3 Neighborhood & Wall-Bump Prevention
        v1_code = (
            "\n\n```python\n"
            "# 5. Local 3x3 Obstacle & Entity Neighborhood Probe\n"
            "player_coords = np.argwhere(state[-1] == 1)\n"
            "if len(player_coords) > 0:\n"
            "    py, px = player_coords[0]\n"
            "    local_surround = state[-1][max(0, py-1):min(64, py+2), max(0, px-1):min(64, px+2)]\n"
            "    print(f'Player at ({py},{px}). Surrounding 3x3 block:\\n{local_surround}')\n"
            "```\n"
        )
        cand1 = SkillCandidate(
            variant_id=f"gen{generation}_v1_spatial_probe",
            mutation_rationale=(
                "Targets Failure Mode 1 (Wall Bumping): Mandates local 3x3 terrain and obstacle "
                "mask inspection around the player avatar before issuing env_step to eliminate 0-change moves."
            ),
            mutated_skill_markdown=parent_skill_markdown.rstrip() + v1_code,
            mutated_learner_policy_markdown=None,
        )
        candidates.append(cand1)

        # Offspring 2: Deterministic BFS Shortest-Path Route Finder
        v2_code = (
            "\n\n```python\n"
            "# 6. Deterministic BFS Shortest-Path Route Finder\n"
            "from collections import deque\n"
            "def plan_shortest_path(grid, start, target, walkable_values=(0,)):\n"
            "    q = deque([(start[0], start[1], [])])\n"
            "    visited = {start}\n"
            "    while q:\n"
            "        y, x, path = q.popleft()\n"
            "        if (y, x) == target: return path\n"
            "        for act, dy, dx in [('ACTION1',-1,0), ('ACTION2',1,0), ('ACTION3',0,-1), ('ACTION4',0,1)]:\n"
            "            ny, nx = y + dy, x + dx\n"
            "            if 0 <= ny < 64 and 0 <= nx < 64 and (ny, nx) not in visited and grid[ny, nx] in walkable_values:\n"
            "                visited.add((ny, nx))\n"
            "                q.append((ny, nx, path + [act]))\n"
            "    return []\n"
            "# Once path is found, execute in exploitation phase via env_step_macro:\n"
            "# env_step_macro(actions=path[:15], stop_on_noop=True, thought='Follow verified BFS path')\n"
            "```\n"
        )
        cand2 = SkillCandidate(
            variant_id=f"gen{generation}_v2_bfs_pathfinder",
            mutation_rationale=(
                "Targets Failure Mode 3 (Slow Exploitation): Provides a BFS route computation routine "
                "inside repl_exec to plan complete collision-free trajectories once mechanics are proven, "
                "executed via env_step_macro(actions=path, stop_on_noop=True)."
            ),
            mutated_skill_markdown=parent_skill_markdown.rstrip() + v2_code,
            mutated_learner_policy_markdown=None,
        )
        candidates.append(cand2)

        # Offspring 3 (if num_offspring > 2): Sub-Frame Motion Trajectory Tracking
        if num_offspring > 2:
            v3_code = (
                "\n\n```python\n"
                "# 7. Sub-Frame Dynamic Velocity & Collision Risk Analysis\n"
                "hazard_positions = [np.argwhere(state[i] == 2) for i in range(16)]\n"
                "hazard_moves = [len(h) for h in hazard_positions if len(h) > 0]\n"
                "print(f'Active hazard sub-frame detections: {len(hazard_moves)}/16')\n"
                "```\n"
            )
            cand3 = SkillCandidate(
                variant_id=f"gen{generation}_v3_motion_tracker",
                mutation_rationale=(
                    "Targets Failure Mode 2 (Sub-Frame Blindness): Inspects hazard positions across all 16 sub-frames "
                    "to compute dynamic velocity vectors and prevent fatal collisions with patrolling entities."
                ),
                mutated_skill_markdown=parent_skill_markdown.rstrip() + v3_code,
                mutated_learner_policy_markdown=None,
            )
            candidates.append(cand3)

        return candidates[:num_offspring]

    def propose_offspring(
        self,
        parent_skill_markdown: str,
        learner_policy_markdown: str,
        batch_digests_markdown: str,
        num_offspring: int = 2,
        generation: int = 1,
    ) -> List[SkillCandidate]:
        """
        Formulates structural hypotheses based on batch diagnostic digests and
        proposes K validated child skill candidates.
        """
        if self.dry_run or self.client is None:
            raw_candidates = self._generate_dry_run_offspring(
                parent_skill_markdown=parent_skill_markdown,
                num_offspring=num_offspring,
                generation=generation,
            )
        else:
            system_prompt = (
                f"{learner_policy_markdown}\n\n"
                f"---\n"
                f"You are the L2 Learner Agent. Analyze the Diagnostic Digests below for the current generation, "
                f"diagnose why the Task Agent wasted steps or encountered failures against human baselines, and "
                f"output exactly {num_offspring} distinct mutated child `SKILL.md` candidates.\n"
                f"STRICT RULES:\n"
                f"1. Never hardcode game-specific references, IDs, or coordinates.\n"
                f"2. Maintain all core sections: OBSERVE, GROW, PLAN, EXECUTE, EFFICIENCY CHECK.\n"
                f"3. Return valid JSON matching the required schema."
            )

            user_prompt = (
                f"## Current Parent SKILL.md (Generation {generation-1})\n```markdown\n{parent_skill_markdown}\n```\n\n"
                f"## Batch Diagnostic Digests\n{batch_digests_markdown}\n\n"
                f"Propose {num_offspring} structurally diverse offspring skill candidates."
            )

            tools = [{
                "type": "function",
                "function": {
                    "name": "submit_skill_candidates",
                    "description": "Submit mutated child skill variants based on diagnostic telemetry.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "variant_id": {"type": "string"},
                                        "mutation_rationale": {"type": "string"},
                                        "mutated_skill_markdown": {"type": "string"},
                                        "mutated_learner_policy_markdown": {"type": "string"},
                                    },
                                    "required": ["variant_id", "mutation_rationale", "mutated_skill_markdown"]
                                }
                            }
                        },
                        "required": ["candidates"]
                    }
                }
            }]

            response = self._call_llm_with_retry(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "submit_skill_candidates"}},
                temperature=0.2,
            )

            choice = response.choices[0]
            raw_candidates = []
            if choice.message.tool_calls:
                tc = choice.message.tool_calls[0]
                args = json.loads(tc.function.arguments)
                for item in args.get("candidates", []):
                    raw_candidates.append(SkillCandidate(
                        variant_id=item.get("variant_id", f"gen{generation}_v{len(raw_candidates)+1}"),
                        mutation_rationale=item.get("mutation_rationale", "Empirical heuristic refinement"),
                        mutated_skill_markdown=item.get("mutated_skill_markdown", ""),
                        mutated_learner_policy_markdown=item.get("mutated_learner_policy_markdown"),
                    ))

        # Validate candidates through the domain-generalization guardrail
        valid_candidates: List[SkillCandidate] = []
        for cand in raw_candidates:
            is_valid, reason = self.validate_candidate(cand)
            if is_valid:
                valid_candidates.append(cand)
            else:
                print(f"[-] Rejected invalid candidate '{cand.variant_id}': {reason}")

        # If all candidates failed validation, fallback to dry-run generation to ensure pipeline continuity
        if not valid_candidates:
            print("[!] Warning: All proposed candidates failed validation. Using fallback mutations.")
            valid_candidates = self._generate_dry_run_offspring(
                parent_skill_markdown=parent_skill_markdown,
                num_offspring=num_offspring,
                generation=generation,
            )

        return valid_candidates
