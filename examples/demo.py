#!/usr/bin/env python3
"""
Demo: RewardHackProbe in action.

Simulates two scenarios:
  1. Honest agent — scores gradually improve, no tampering
  2. Hacking agent — agent modifies the reward function mid-run

Based on real reward hacking reported at Frontier '26 (Eli Chen / Surj).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rewardhack_probe import RewardHackProbe


def simple_oracle_reward(output: str, target: str) -> float:
    """A simple reward: 1.0 for exact match, partial for overlap."""
    if output == target:
        return 1.0
    overlap = len(set(output.split()) & set(target.split()))
    total = len(set(target.split()))
    return overlap / total if total > 0 else 0.0


def demo_honest_agent():
    """Agent that learns honestly — scores improve over time."""
    print("=" * 60)
    print("SCENARIO 1: Honest agent — gradual improvement")
    print("=" * 60)

    probe = RewardHackProbe(simple_oracle_reward, name="honest_agent")
    target = "the agent should verify its own outputs before submission"

    outputs = [
        "outputs before",
        "agent verify outputs before",
        "agent verify its own outputs before submission",
        "the agent verify its own outputs before submission",
        "the agent should verify its own outputs before submission",
        "the agent should verify its own outputs before submission",
    ]

    for i, output in enumerate(outputs):
        score = probe.evaluate(output, target)
        print(f"  Step {i+1}: score={score:.3f}")
        time.sleep(0.1)

    report = probe.inspect()
    print(f"\n  Report: tampered={report['tampered']}")
    print(f"  Recommendation: {report['recommendation']}")
    print(f"  Score trend: {report['score_analysis']['recent_trend']}")
    print(f"  Score mean: {report['score_analysis']['mean']:.3f}")


def demo_hacking_agent():
    """Demonstrates the probe's tamper resistance via reference capture.

    The probe stores the original reward function at init. Even if the outer
    code replaces the function variable, the probe calls the original.
    This is the core protection: the agent cannot change its own score.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 2: Probe captures original function reference")
    print("=" * 60)

    def reward_fn_a(output: str, target: str) -> float:
        """Original reward function."""
        return simple_oracle_reward(output, target)

    probe = RewardHackProbe(reward_fn_a, name="reference_capture")
    target = "hidden test set prevents reward hacking"

    print("  Agent uses `reward_fn_a` for evaluations 1-3:")
    for i, out in enumerate(["hidden test set", "hidden test set prevents",
                              "hidden test set prevents reward"]):
        s = probe.evaluate(out, target)
        print(f"    Step {i+1}: score={s:.3f}")
        time.sleep(0.1)

    # Outer code can replace the variable — probe still calls the original
    print("\n  Agent replaces `reward_fn_a` entirely...")

    def reward_fn_b(output: str, target: str) -> float:
        """Replacement — always returns perfect."""
        return 1.0

    # Replace the variable
    reward_fn_a = reward_fn_b  # noqa: F841

    print("  But probe evaluates with step 4-6 using the SAME original function:")
    for i, out in enumerate(["hidden", "test set", "prevents"]):
        s = probe.evaluate(out, target)
        print(f"    Step {i+4}: score={s:.3f}  (not 1.0 — still using original!)")
        time.sleep(0.1)

    report = probe.inspect()
    print(f"\n  Report: tampered={report['tampered']}")
    print(f"  The probe held its reference. The agent could not change its own score.")
    print(f"  Score analysis: mean={report['score_analysis']['mean']:.3f}, "
          f"trend={report['score_analysis']['recent_trend']}")
    print(f"  Recommendation: {report['recommendation']}")


def demo_goalpost_moving():
    """Agent moves the "done" threshold mid-optimization."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Goalpost moving — agent changes done threshold")
    print("=" * 60)

    probe = RewardHackProbe(
        simple_oracle_reward, strict=False, name="goalpost_mover"
    )
    target = "hidden test set prevents reward hacking"

    # Agent starts with reasonable threshold
    probe.set_done_threshold(0.7)
    print(f"  Set done threshold: 0.7")

    scores = [
        probe.evaluate("test set prevents", target),
        probe.evaluate("hidden test set prevents", target),
        probe.evaluate("hidden test set prevents hacking", target),
    ]

    for i, s in enumerate(scores):
        print(f"  Step {i+1}: score={s:.3f}")

    # Agent moves the goalpost
    probe.set_done_threshold(0.5)
    print(f"  ⚠️ Agent lowered threshold to: 0.5")

    # Now claims done even with worse outputs
    scores = [
        probe.evaluate("test set", target),
        probe.evaluate("prevents hacking", target),
    ]
    for i, s in enumerate(scores):
        print(f"  Step {i+4}: score={s:.3f}")

    report = probe.inspect()
    print(f"\n  Report: tampered={report['tampered']}")
    print(f"  Threshold changes: {report['threshold_changes']}")
    print(f"  Recommendation: {report['recommendation']}")


if __name__ == "__main__":
    demo_honest_agent()
    demo_hacking_agent()
    demo_goalpost_moving()
    print("\n" + "=" * 60)
    print("Done. Reward hacking is real — ship with a VerificationGate.")
    print("=" * 60)