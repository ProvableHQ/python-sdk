---
name: shield-swap
description: Use when the user wants to trade, LP, or build on the shield_swap AMM — setting up an account, redeeming an invite, getting the airdrop, swapping privately, managing liquidity positions, or collecting earnings via the aleo_shield_swap Python SDK.
---

Read the packaged agent guide (generated from the SDK — always matches the
installed version):

    python -m aleo_shield_swap

Follow its Tier 1 lifecycle and conversation pattern. Write Python against
the SDK; don't re-implement flows the verbs already provide, and don't read
SDK source unless the guide genuinely lacks the answer. Preconditions are
enforced in code — on error, read the exception message; it names the verb
that fixes it.

To install this skill into a Claude Code project, copy this directory to
`.claude/skills/shield-swap/` — or just point the agent at the guide above.
