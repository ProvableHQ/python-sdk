program arc22_token.aleo;

record Token:
    owner as address.private;
    amount as u128.private;
    token_id as field.private;

record ComplianceRecord:
    owner as address.private;
    amount as u128.private;
    sender as address.private;
    recipient as address.private;

struct MerkleProof:
    siblings as [field; 16u32];
    leaf_index as u32;

struct approval:
    approver as address;
    spender as address;

mapping account:
    key as address.public;
    value as u128.public;

mapping approvals:
    key as field.public;
    value as u128.public;

mapping settings:
    key as u8.public;
    value as u128.public;

mapping token_info:
    key as u8.public;
    value as u8.public;

function approve_public:
    input r0 as address.public;
    input r1 as u128.public;
    async approve_public self.caller r0 r1 into r2;
    output r2 as arc22_token.aleo/approve_public.future;

finalize approve_public:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u128.public;
    cast r0 r1 into r3 as approval;
    hash.bhp256 r3 into r4 as field;
    get.or_use approvals[r4] 0u128 into r5;
    add r5 r2 into r6;
    set r6 into approvals[r4];

function unapprove_public:
    input r0 as address.public;
    input r1 as u128.public;
    async unapprove_public self.caller r0 r1 into r2;
    output r2 as arc22_token.aleo/unapprove_public.future;

finalize unapprove_public:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u128.public;
    cast r0 r1 into r3 as approval;
    hash.bhp256 r3 into r4 as field;
    get approvals[r4] into r5;
    sub r5 r2 into r6;
    set r6 into approvals[r4];

function transfer_public:
    input r0 as address.public;
    input r1 as u128.public;
    async transfer_public self.caller r0 r1 into r2;
    output r2 as arc22_token.aleo/transfer_public.future;

finalize transfer_public:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u128.public;
    get account[r0] into r3;
    sub r3 r2 into r4;
    set r4 into account[r0];
    get.or_use account[r1] 0u128 into r5;
    add r5 r2 into r6;
    set r6 into account[r1];

function transfer_private:
    input r0 as address.private;
    input r1 as u128.private;
    input r2 as Token.record;
    input r3 as [MerkleProof; 2u32].private;
    sub r2.amount r1 into r4;
    cast r2.owner r1 r2.owner r0 into r5 as ComplianceRecord.record;
    cast r2.owner r4 r2.token_id into r6 as Token.record;
    cast r0 r1 r2.token_id into r7 as Token.record;
    async transfer_private r0 r1 into r8;
    output r5 as ComplianceRecord.record;
    output r6 as Token.record;
    output r7 as Token.record;
    output r8 as arc22_token.aleo/transfer_private.future;

finalize transfer_private:
    input r0 as address.public;
    input r1 as u128.public;
    get.or_use account[r0] 0u128 into r2;
    add r2 r1 into r3;
    set r3 into account[r0];

function transfer_private_to_public:
    input r0 as address.public;
    input r1 as u128.public;
    input r2 as Token.record;
    input r3 as [MerkleProof; 2u32].private;
    sub r2.amount r1 into r4;
    cast r2.owner r1 r2.owner r0 into r5 as ComplianceRecord.record;
    cast r2.owner r4 r2.token_id into r6 as Token.record;
    async transfer_private_to_public r0 r1 into r7;
    output r5 as ComplianceRecord.record;
    output r6 as Token.record;
    output r7 as arc22_token.aleo/transfer_private_to_public.future;

finalize transfer_private_to_public:
    input r0 as address.public;
    input r1 as u128.public;
    get.or_use account[r0] 0u128 into r2;
    add r2 r1 into r3;
    set r3 into account[r0];

function transfer_public_to_private:
    input r0 as address.private;
    input r1 as u128.public;
    cast r0 r1 self.caller r0 into r2 as ComplianceRecord.record;
    cast r0 r1 0field into r3 as Token.record;
    async transfer_public_to_private self.caller r1 into r4;
    output r2 as ComplianceRecord.record;
    output r3 as Token.record;
    output r4 as arc22_token.aleo/transfer_public_to_private.future;

finalize transfer_public_to_private:
    input r0 as address.public;
    input r1 as u128.public;
    get account[r0] into r2;
    sub r2 r1 into r3;
    set r3 into account[r0];

function transfer_from_public:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u128.public;
    async transfer_from_public self.caller r0 r1 r2 into r3;
    output r3 as arc22_token.aleo/transfer_from_public.future;

finalize transfer_from_public:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as address.public;
    input r3 as u128.public;
    cast r1 r0 into r4 as approval;
    hash.bhp256 r4 into r5 as field;
    get approvals[r5] into r6;
    sub r6 r3 into r7;
    set r7 into approvals[r5];
    get account[r1] into r8;
    sub r8 r3 into r9;
    set r9 into account[r1];
    get.or_use account[r2] 0u128 into r10;
    add r10 r3 into r11;
    set r11 into account[r2];

function transfer_from_public_to_private:
    input r0 as address.public;
    input r1 as address.private;
    input r2 as u128.public;
    cast r1 r2 r0 r1 into r3 as ComplianceRecord.record;
    cast r1 r2 0field into r4 as Token.record;
    async transfer_from_public_to_private self.caller r0 r2 into r5;
    output r3 as ComplianceRecord.record;
    output r4 as Token.record;
    output r5 as arc22_token.aleo/transfer_from_public_to_private.future;

finalize transfer_from_public_to_private:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u128.public;
    cast r1 r0 into r3 as approval;
    hash.bhp256 r3 into r4 as field;
    get approvals[r4] into r5;
    sub r5 r2 into r6;
    set r6 into approvals[r4];
    get account[r1] into r7;
    sub r7 r2 into r8;
    set r8 into account[r1];

function transfer_public_as_signer:
    input r0 as address.public;
    input r1 as u128.public;
    async transfer_public_as_signer self.signer r0 r1 into r2;
    output r2 as arc22_token.aleo/transfer_public_as_signer.future;

finalize transfer_public_as_signer:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u128.public;
    get account[r0] into r3;
    sub r3 r2 into r4;
    set r4 into account[r0];
    get.or_use account[r1] 0u128 into r5;
    add r5 r2 into r6;
    set r6 into account[r1];

function join:
    input r0 as Token.record;
    input r1 as Token.record;
    add r0.amount r1.amount into r2;
    cast r0.owner r2 r0.token_id into r3 as Token.record;
    output r3 as Token.record;

function split:
    input r0 as Token.record;
    input r1 as u128.private;
    sub r0.amount r1 into r2;
    cast r0.owner r1 r0.token_id into r3 as Token.record;
    cast r0.owner r2 r0.token_id into r4 as Token.record;
    output r3 as Token.record;
    output r4 as Token.record;

view balance_of:
    input r0 as address.public;
    get.or_use account[r0] 0u128 into r1;
    output r1 as u128.public;

view allowance:
    input r0 as address.public;
    input r1 as address.public;
    cast r0 r1 into r2 as approval;
    hash.bhp256 r2 into r3 as field;
    get.or_use approvals[r3] 0u128 into r4;
    output r4 as u128.public;

view supply:
    get.or_use settings[0u8] 0u128 into r0;
    output r0 as u128.public;

view max_supply:
    get.or_use settings[1u8] 0u128 into r0;
    output r0 as u128.public;

view decimals:
    get.or_use token_info[0u8] 6u8 into r0;
    output r0 as u8.public;

view name:
    cast 6577149field into r0 as identifier;
    output r0 as identifier.public;

view symbol:
    cast 5526356field into r0 as identifier;
    output r0 as identifier.public;
