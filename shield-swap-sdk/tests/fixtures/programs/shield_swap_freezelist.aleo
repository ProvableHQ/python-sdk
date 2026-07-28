import shield_swap_multisig_core.aleo;

program shield_swap_freezelist.aleo;

struct ChecksumEdition:
    checksum as [u8; 32u32];
    edition as u16;

mapping freeze_list:
    key as address.public;
    value as boolean.public;

mapping freeze_list_index:
    key as u32.public;
    value as address.public;

mapping freeze_list_last_index:
    key as boolean.public;
    value as u32.public;

mapping freeze_list_root:
    key as u8.public;
    value as field.public;

mapping root_updated_height:
    key as boolean.public;
    value as u32.public;

mapping block_height_window:
    key as boolean.public;
    value as u32.public;

mapping previous_root_window:
    key as boolean.public;
    value as u32.public;

mapping address_to_role:
    key as address.public;
    value as u16.public;

function initialize:
    input r0 as address.public;
    input r1 as u32.public;
    async initialize r0 self.caller r1 into r2;
    output r2 as shield_swap_freezelist.aleo/initialize.future;

finalize initialize:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u32.public;
    contains freeze_list_root[1u8] into r3;
    not r3 into r4;
    assert.eq r4 true;
    assert.eq r1 aleo1z3zwzgpgakk89xpknync5rtklkjkyv33g7cvaqe0gku64zs3lv9qyux0qc;
    set 8u16 into address_to_role[r0];
    set r2 into block_height_window[true];
    set 0u32 into freeze_list_last_index[true];
    set 3642222252059314292809609689035560016959342421640560347114299934615987159853field into freeze_list_root[1u8];
    set false into freeze_list[aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc];
    set aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into freeze_list_index[0u32];

function update_role:
    input r0 as address.public;
    input r1 as u16.public;
    async update_role r0 self.caller r1 into r2;
    output r2 as shield_swap_freezelist.aleo/update_role.future;

finalize update_role:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as u16.public;
    get address_to_role[r1] into r3;
    and r3 8u16 into r4;
    is.eq r4 8u16 into r5;
    assert.eq r5 true;
    is.eq r1 r0 into r6;
    branch.eq r6 false to end_then_0_0;
    and r2 8u16 into r7;
    is.eq r7 8u16 into r8;
    assert.eq r8 true;
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;
    set r2 into address_to_role[r0];

function update_block_height_window:
    input r0 as u32.public;
    async update_block_height_window r0 self.caller into r1;
    output r1 as shield_swap_freezelist.aleo/update_block_height_window.future;

finalize update_block_height_window:
    input r0 as u32.public;
    input r1 as address.public;
    get address_to_role[r1] into r2;
    and r2 16u16 into r3;
    is.eq r3 16u16 into r4;
    assert.eq r4 true;
    set r0 into block_height_window[true];

function update_freeze_list:
    input r0 as address.public;
    input r1 as boolean.public;
    input r2 as u32.public;
    input r3 as field.public;
    input r4 as field.public;
    async update_freeze_list r0 r1 r2 self.caller r3 r4 into r5;
    output r5 as shield_swap_freezelist.aleo/update_freeze_list.future;

finalize update_freeze_list:
    input r0 as address.public;
    input r1 as boolean.public;
    input r2 as u32.public;
    input r3 as address.public;
    input r4 as field.public;
    input r5 as field.public;
    get address_to_role[r3] into r6;
    and r6 16u16 into r7;
    is.eq r7 16u16 into r8;
    assert.eq r8 true;
    assert.neq r2 0u32;
    assert.neq r0 aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc;
    get freeze_list_root[1u8] into r9;
    assert.eq r4 r9;
    assert.neq r5 r9;
    set r9 into freeze_list_root[2u8];
    set r5 into freeze_list_root[1u8];
    get.or_use freeze_list[r0] false into r10;
    assert.neq r1 r10;
    set r1 into freeze_list[r0];
    get.or_use freeze_list_index[r2] aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into r11;
    branch.eq r1 false to end_then_0_0;
    assert.eq r11 aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc;
    get freeze_list_last_index[true] into r12;
    add r12 1u32 into r13;
    gte r13 r2 into r14;
    assert.eq r14 true;
    lt r12 r2 into r15;
    branch.eq r15 false to end_then_1_2;
    lt r12 32768u32 into r16;
    assert.eq r16 true;
    set r2 into freeze_list_last_index[true];
    branch.eq true true to end_otherwise_1_3;
    position end_then_1_2;
    position end_otherwise_1_3;
    set r0 into freeze_list_index[r2];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    assert.eq r11 r0;
    set aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into freeze_list_index[r2];
    position end_otherwise_0_1;
    get block_height_window[true] into r17;
    set r17 into previous_root_window[true];
    set block.height into root_updated_height[true];

constructor:
    is.eq edition 0u16 into r0;
    branch.eq r0 false to end_then_0_0;
    assert.eq program_owner aleo1z3zwzgpgakk89xpknync5rtklkjkyv33g7cvaqe0gku64zs3lv9qyux0qc;
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;
    gt edition 0u16 into r1;
    branch.eq r1 false to end_then_0_2;
    cast checksum edition into r2 as ChecksumEdition;
    hash.bhp256 r2 into r3 as field;
    cast shield_swap_freezelist.aleo r3 into r4 as shield_swap_multisig_core.aleo/WalletSigningOpId;
    hash.bhp256 r4 into r5 as field;
    contains shield_swap_multisig_core.aleo/completed_signing_ops[r5] into r6;
    assert.eq r6 true;
    branch.eq true true to end_otherwise_0_3;
    position end_then_0_2;
    position end_otherwise_0_3;

