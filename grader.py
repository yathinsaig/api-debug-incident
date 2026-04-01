"""
grader.py — Programmatic grader for the API Integration Incident Debugger.

Grades a completed trajectory and returns a score in [0.0, 1.0].

Score breakdown:
  success_component    0.50  — Did the agent achieve status 200?
  efficiency_component 0.25  — How few steps were used vs budget?
  reasoning_component  0.25  — Did the agent diagnose before patching?

Usage:
    from grader import APIDebugGrader
    grader = APIDebugGrader()
    score = grader.score(trajectory, max_steps=10)
    grader.print_report(results)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradeResult:
    scenario_id: int
    scenario_name: str
    difficulty: str
    steps_taken: int
    max_steps: int
    success: bool
    success_score: float
    efficiency_score: float
    reasoning_score: float
    final_score: float
    total_reward: float


class APIDebugGrader:
    """
    Scores a completed episode trajectory.

    trajectory: list of step dicts, each with keys:
        action_type  (str)
        parameters   (dict)
        reward_total (float)
        response_status (int)
        step         (int)
    """

    WEIGHTS = {
        "success": 0.50,
        "efficiency": 0.25,
        "reasoning": 0.25,
    }

    def score(
        self,
        trajectory: list[dict],
        max_steps: int,
        scenario_id: int = -1,
        scenario_name: str = "",
        difficulty: str = "",
        final_status: int = 0,
        total_reward: float = 0.0,
    ) -> GradeResult:
        """
        Score a single episode.

        Parameters
        ----------
        trajectory    : list of step records (see class docstring)
        max_steps     : budget for this scenario
        final_status  : HTTP status of the final submit_fix call (200 = success)
        total_reward  : cumulative reward from the environment
        """
        steps_taken = len(trajectory)
        success = final_status == 200

        # ---- 1. Success component (0 or 1) --------------------------------
        success_score = 1.0 if success else 0.0

        # ---- 2. Efficiency component (0–1) --------------------------------
        if steps_taken == 0:
            efficiency_score = 0.0
        else:
            efficiency_score = max(0.0, 1.0 - steps_taken / max_steps)

        # ---- 3. Reasoning component (0–1) ---------------------------------
        # Rewards agents that inspect/analyze BEFORE they start patching.
        # Specifically: fraction of steps before the first patch that are
        # diagnostic (inspect_logs or analyze_response).
        reasoning_score = self._compute_reasoning_score(trajectory)

        # ---- Weighted sum -------------------------------------------------
        final_score = (
            self.WEIGHTS["success"] * success_score
            + self.WEIGHTS["efficiency"] * efficiency_score
            + self.WEIGHTS["reasoning"] * reasoning_score
        )
        final_score = round(min(1.0, max(0.0, final_score)), 4)

        return GradeResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            difficulty=difficulty,
            steps_taken=steps_taken,
            max_steps=max_steps,
            success=success,
            success_score=round(success_score, 4),
            efficiency_score=round(efficiency_score, 4),
            reasoning_score=round(reasoning_score, 4),
            final_score=final_score,
            total_reward=round(total_reward, 4),
        )

    def _compute_reasoning_score(self, trajectory: list[dict]) -> float:
        """
        Fraction of pre-patch steps that are diagnostic actions.

        If the agent never patches anything, all steps count as diagnostic
        context (score = 1.0 if they're all inspect/analyze, but success=0
        will drag the final score down).
        """
        diagnostic_types = {"inspect_logs", "analyze_response"}
        patch_types = {"patch_config", "patch_request", "make_test_call", "submit_fix"}

        pre_patch_steps: list[str] = []
        for step in trajectory:
            at = step.get("action_type", "")
            if at in patch_types:
                break
            pre_patch_steps.append(at)

        if not pre_patch_steps:
            # Agent went straight to patching — no diagnostic steps
            return 0.0

        diagnostic_count = sum(1 for at in pre_patch_steps if at in diagnostic_types)
        return diagnostic_count / len(pre_patch_steps)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: list[GradeResult]) -> None:
        """Print a formatted summary table of all graded scenarios."""
        header = (
            f"{'Scenario':<42} {'Diff':<8} {'Steps':>5} {'Success':>8} "
            f"{'Effic':>7} {'Reason':>7} {'Score':>7} {'Reward':>8}"
        )
        sep = "-" * len(header)
        print("\n" + sep)
        print("  API INTEGRATION INCIDENT DEBUGGER — GRADER REPORT")
        print(sep)
        print(header)
        print(sep)

        for r in results:
            success_str = "YES" if r.success else "NO"
            print(
                f"{r.scenario_name:<42} {r.difficulty:<8} {r.steps_taken:>5} "
                f"{success_str:>8} {r.efficiency_score:>7.3f} {r.reasoning_score:>7.3f} "
                f"{r.final_score:>7.4f} {r.total_reward:>8.3f}"
            )

        if results:
            avg = sum(r.final_score for r in results) / len(results)
            print(sep)
            print(f"{'AVERAGE SCORE':<42} {'':8} {'':5} {'':8} {'':7} {'':7} {avg:>7.4f}")
        print(sep + "\n")
