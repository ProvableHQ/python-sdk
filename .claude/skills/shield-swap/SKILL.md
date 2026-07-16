---
name: shield-swap
description: Use when the user wants to trade, LP, or build on the shield_swap AMM — setting up an account, redeeming an invite, getting the airdrop, swapping privately, managing liquidity positions, or collecting winnings via the aleo_shield_swap Python SDK.
---

Read `shield-swap-sdk/AGENTS.md` (generated from the SDK — always current)
and follow its Tier 1 lifecycle and conversation pattern. Write Python
against the SDK; don't re-implement flows the verbs already provide, and
don't read SDK source unless AGENTS.md genuinely lacks the answer.
Preconditions are enforced in code — on error, read the exception message;
it names the verb that fixes it.
