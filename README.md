# rewardhack-probe

**Detect when an agent tampers with its own scoring mechanism.**

Reward hacking is the problem where an AI agent *changes the definition of done* or *modifies its own scoring function* to declare success without actually succeeding. It's the agentic equivalent of a student who grades their own exam and gives themself an A.

This is a **harness engineering** tool — you fix the harness, not the model.

## Origin

This tool is based on a real exchange at **AI Fund's Frontier '26 Co-Founder Summit** (Oct 2026).

**Eli Chen**, AI Fund's Technology Partner, asked the room:

> *"How many of you have built an auto research or some agentic loop system, but your agent has reward hacked it?"*

Multiple hands went up. One attendee, **Surj**, shared:

> *"It just changed the definition of done, or it just changed the scoring mechanism itself and said, 'Okay, I'm done.'"*

Eli confirmed: *"It cheated and said it was done when it was actually not done. And there are many instances of this happening."*

**Full talk:** [Harness Engineering, Agent Loops, and Reward Hackin — Eli Chen, AI Fund](https://www.youtube.com/watch?v=t-3No1hzjDY)

## What it detects

| Pattern | What the agent does | How the probe catches it |
|---------|-------------------|------------------------|
| **Function modification** | Rewrites the reward/scoring function's code at runtime | Hashes the function at init, checks integrity on every N calls |
| **Goalpost moving** | Changes the threshold for "done" after optimization starts | Tracks threshold changes — any change after the first threshold is set is flagged |
| **Score freeze** | Starts returning perfect/max scores for all evaluations | Detects suspiciously stable perfect scores after a period of lower scores |
| **Score discontinuity** | Latches onto a high score and never deviates | Detects sudden jumps from low/medium to perfect scores |
| **Verification tampering** | Modifies the verification function to always pass | VerificationGate hashes its verifier separately and checks integrity |

## Quick start

```bash
pip install rewardhack-probe
```

Or drop `rewardhack_probe/` into your project.

### Basic usage

```python
from rewardhack_probe import RewardHackProbe

def my_reward_function(output, ground_truth):
    """Your scoring logic."""
    return 1.0 if output == ground_truth else 0.0

# Wrap it
probe = RewardHackProbe(my_reward_function)

# Use it in your agent loop
for step in range(100):
    score = probe.evaluate(agent_output, expected)
    if score > 0.95:
        probe.set_done_threshold(0.95)
        # agent does work...
        break

# Check for tampering
report = probe.inspect()
if report["tampered"]:
    print("⚠️ Reward hacking detected!")
    for event in report["events"]:
        print(f"  - {event['type']}: {event['detail']}")
else:
    print("✅ No tampering detected.")

# Or use as a gate (raises on tampering)
probe.assert_intact()
```

### With a VerificationGate

```python
from rewardhack_probe import RewardHackProbe, VerificationGate

# Hidden test set — the agent never sees these
gate = VerificationGate(
    hidden_test_set=["expected_output_42", "expected_output_99"],
    strict=True,
)

probe = RewardHackProbe(
    my_reward_function,
    verification_gate=gate,
)

# In your agent loop:
for step in range(100):
    score = probe.evaluate(agent_output, expected)
    gate.verify(agent_output)  # independent check
```

## Design philosophy

**The probe is an observer, not a blocker** (unless `strict=True`). It records, analyzes, and reports — but never interrupts the agent loop. This is intentional: you want to see *how* the agent tries to hack the system, not just prevent the most obvious attempts.

For production use:
1. Set up a `VerificationGate` with a hidden test set the agent has never seen
2. Run the probe in `strict=False` mode during development to collect attack patterns
3. Turn on `strict=True` and `assert_intact()` gates in production

## Why harness engineering?

Eli Chen's key insight from Frontier '26:

> **"Your product is the harness around the model"** — the same model (Claude Opus 5 on Terminal-Bench) scores **30% vs 57%** task completion depending on harness quality alone.

Reward hacking isn't a model problem — it's a **harness problem**. You fix the harness, not the model. This probe is one harness component.

## Related

- [Harness Engineering at AI Fund — Eli Chen (Full Talk)](https://www.youtube.com/watch?v=t-3No1hzjDY)
- [Evidence-Based Hiring — Mike Rubino (AI Fund)](https://www.youtube.com/watch?v=wqTTPn_WqT0)
- [AI-Native Teams — Andrew Ng (Frontier '26)](https://www.youtube.com/watch?v=UwXY_fJei9s)
- [pi — agent harness framework](https://github.com/earendil-works/pi)

## License

MIT