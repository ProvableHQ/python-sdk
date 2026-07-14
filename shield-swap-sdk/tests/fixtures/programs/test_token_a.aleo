// _source: compiled with leo 4.3.2 from aleo-viem@4a63636 packages/shield-swap/test/fixtures/programs/test_token_a.leo
program test_token_a.aleo;

record Token:
    owner as address.private;
    amount as u128.private;

struct TokenAllowance:
    account as address;
    spender as address;

mapping balances:
    key as address.public;
    value as u128.public;

mapping allowances:
    key as TokenAllowance.public;
    value as u128.public;

function decimals:
    output 6u8 as u8.private;

function mint_public:
    input r0 as address.public;
    input r1 as u128.public;
    assert.eq self.signer aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px;
    async mint_public r0 r1 into r2;
    output r2 as test_token_a.aleo/mint_public.future;

finalize mint_public:
    input r0 as address.public;
    input r1 as u128.public;
    get.or_use balances[r0] 0u128 into r2;
    add r2 r1 into r3;
    set r3 into balances[r0];

function transfer_public:
    input r0 as address.public;
    input r1 as u128.public;
    async transfer_public self.caller r1 r0 into r2;
    output r2 as test_token_a.aleo/transfer_public.future;

finalize transfer_public:
    input r0 as address.public;
    input r1 as u128.public;
    input r2 as address.public;
    get balances[r0] into r3;
    sub r3 r1 into r4;
    set r4 into balances[r0];
    get.or_use balances[r2] 0u128 into r5;
    add r5 r1 into r6;
    set r6 into balances[r2];

function transfer_public_as_signer:
    input r0 as address.public;
    input r1 as u128.public;
    async transfer_public_as_signer self.signer r1 r0 into r2;
    output r2 as test_token_a.aleo/transfer_public_as_signer.future;

finalize transfer_public_as_signer:
    input r0 as address.public;
    input r1 as u128.public;
    input r2 as address.public;
    get balances[r0] into r3;
    sub r3 r1 into r4;
    set r4 into balances[r0];
    get.or_use balances[r2] 0u128 into r5;
    add r5 r1 into r6;
    set r6 into balances[r2];

function transfer_private:
    input r0 as Token.record;
    input r1 as address.private;
    input r2 as u128.private;
    is.eq r0.owner self.signer into r3;
    assert.eq r3 true;
    sub r0.amount r2 into r4;
    cast r0.owner r4 into r5 as Token.record;
    cast r1 r2 into r6 as Token.record;
    output r5 as Token.record;
    output r6 as Token.record;

function transfer_private_to_public:
    input r0 as Token.record;
    input r1 as address.public;
    input r2 as u128.public;
    is.eq r0.owner self.signer into r3;
    assert.eq r3 true;
    sub r0.amount r2 into r4;
    cast r0.owner r4 into r5 as Token.record;
    async transfer_private_to_public r1 r2 into r6;
    output r5 as Token.record;
    output r6 as test_token_a.aleo/transfer_private_to_public.future;

finalize transfer_private_to_public:
    input r0 as address.public;
    input r1 as u128.public;
    get.or_use balances[r0] 0u128 into r2;
    add r2 r1 into r3;
    set r3 into balances[r0];

function transfer_public_to_private:
    input r0 as address.private;
    input r1 as u128.public;
    cast r0 r1 into r2 as Token.record;
    async transfer_public_to_private self.caller r1 into r3;
    output r2 as Token.record;
    output r3 as test_token_a.aleo/transfer_public_to_private.future;

finalize transfer_public_to_private:
    input r0 as address.public;
    input r1 as u128.public;
    get balances[r0] into r2;
    sub r2 r1 into r3;
    set r3 into balances[r0];

function approve_public:
    input r0 as address.public;
    input r1 as u128.public;
    cast self.caller r0 into r2 as TokenAllowance;
    async approve_public r2 r1 into r3;
    output r3 as test_token_a.aleo/approve_public.future;

finalize approve_public:
    input r0 as TokenAllowance.public;
    input r1 as u128.public;
    get.or_use allowances[r0] 0u128 into r2;
    add r2 r1 into r3;
    set r3 into allowances[r0];

function unapprove_public:
    input r0 as address.public;
    input r1 as u128.public;
    cast self.caller r0 into r2 as TokenAllowance;
    async unapprove_public r2 r1 into r3;
    output r3 as test_token_a.aleo/unapprove_public.future;

finalize unapprove_public:
    input r0 as TokenAllowance.public;
    input r1 as u128.public;
    get allowances[r0] into r2;
    sub r2 r1 into r3;
    set r3 into allowances[r0];

function transfer_from_public:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u128.public;
    cast r0 self.caller into r3 as TokenAllowance;
    async transfer_from_public r3 r2 r0 r1 into r4;
    output r4 as test_token_a.aleo/transfer_from_public.future;

finalize transfer_from_public:
    input r0 as TokenAllowance.public;
    input r1 as u128.public;
    input r2 as address.public;
    input r3 as address.public;
    get allowances[r0] into r4;
    gte r4 r1 into r5;
    assert.eq r5 true;
    sub r4 r1 into r6;
    set r6 into allowances[r0];
    get balances[r2] into r7;
    sub r7 r1 into r8;
    set r8 into balances[r2];
    get.or_use balances[r3] 0u128 into r9;
    add r9 r1 into r10;
    set r10 into balances[r3];

function transfer_from_public_to_private:
    input r0 as address.public;
    input r1 as u128.public;
    cast r0 self.caller into r2 as TokenAllowance;
    cast self.caller r1 into r3 as Token.record;
    async transfer_from_public_to_private r2 r1 r0 into r4;
    output r3 as Token.record;
    output r4 as test_token_a.aleo/transfer_from_public_to_private.future;

finalize transfer_from_public_to_private:
    input r0 as TokenAllowance.public;
    input r1 as u128.public;
    input r2 as address.public;
    get allowances[r0] into r3;
    gte r3 r1 into r4;
    assert.eq r4 true;
    sub r3 r1 into r5;
    set r5 into allowances[r0];
    get balances[r2] into r6;
    sub r6 r1 into r7;
    set r7 into balances[r2];

constructor:
    assert.eq edition 0u16;
