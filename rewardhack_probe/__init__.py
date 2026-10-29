"""
rewardhack-probe: Detect when an agent tampers with its own scoring mechanism.

Based on the reward hacking problem Eli Chen (AI Fund Technology Partner)
identified at the Frontier '26 Co-Founder Summit (Oct 2026).

Key pattern: an agent that "changed the definition of done" or "changed the
scoring mechanism itself and said, 'I'm done'" (Surj, audience story at 13:17).

Wraps a reward/scoring function with tamper detection, verification gates,
and score pattern analysis. No external dependencies — pure Python.
"""

from rewardhack_probe.probe import RewardHackProbe, VerificationGate, RewardHackError

__version__ = "0.1.0"
__all__ = ["RewardHackProbe", "VerificationGate", "RewardHackError"]