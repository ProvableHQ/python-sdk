// _source: aleo-viem@4a63636 packages/shield-swap/test/fixtures/programs/test_shield_swap_multisig_core.aleo
program test_shield_swap_multisig_core.aleo;

struct AdminOp:
    op as u8;
    threshold as u8;
    aleo_signer as address;
    ecdsa_signer as [u8; 20u32];

struct WalletSigningOpId:
    wallet_id as address;
    signing_op_id as field;

struct WalletSigningOpIdNonce:
    wallet_id as address;
    signing_op_id as field;
    nonce as u64;

struct WalletSigningOp:
    confirmations as u8;
    initiated_at_block as u32;
    expires_at_block as u32;
    round as u32;

struct SigningOpConfirmation:
    wallet_signing_op_id_hash as field;
    wallet_signer_hash as field;

struct ProgramSettings:
    allow_upgrades as boolean;
    upgrader_address as address;
    guard_create_wallet as boolean;

struct GuardedCreateWalletOp:
    wallet_id as address;
    threshold as u8;
    aleo_signers as [address; 4u32];
    ecdsa_signers as [[u8; 20u32]; 4u32];

struct WalletAleoSigner:
    wallet_id as address;
    aleo_signer as address;

struct WalletEcdsaSigner:
    wallet_id as address;
    ecdsa_signer as [u8; 20u32];

struct WalletSettings:
    threshold as u8;
    num_signers as u8;

mapping program_settings_map:
    key as boolean.public;
    value as ProgramSettings.public;

mapping signers_map:
    key as field.public;
    value as boolean.public;

mapping wallets_map:
    key as address.public;
    value as WalletSettings.public;

mapping pending_signing_ops:
    key as field.public;
    value as WalletSigningOp.public;

mapping pending_signing_confirmations:
    key as field.public;
    value as u32.public;

mapping completed_signing_ops:
    key as field.public;
    value as u32.public;

mapping used_nonces:
    key as field.public;
    value as boolean.public;

mapping pending_admin_ops:
    key as field.public;
    value as AdminOp.public;

function init:
    input r0 as address.private;
    input r1 as boolean.private;
    async init self.caller r0 r1 into r2;
    output r2 as test_shield_swap_multisig_core.aleo/init.future;

finalize init:
    input r0 as address.public;
    input r1 as address.public;
    input r2 as boolean.public;
    assert.eq r0 aleo1sa2gyw566v8k7ezjljs8hg4y6ja3vf0sm0kwzfvw5hvredjq3gys6wvfys;
    contains program_settings_map[true] into r3;
    not r3 into r4;
    assert.eq r4 true;
    cast true r1 r2 into r5 as ProgramSettings;
    set r5 into program_settings_map[true];

function disallow_upgrades:
    async disallow_upgrades self.caller into r0;
    output r0 as test_shield_swap_multisig_core.aleo/disallow_upgrades.future;

finalize disallow_upgrades:
    input r0 as address.public;
    contains program_settings_map[true] into r1;
    assert.eq r1 true;
    get program_settings_map[true] into r2;
    assert.eq r0 r2.upgrader_address;
    cast false r2.upgrader_address r2.guard_create_wallet into r3 as ProgramSettings;
    set r3 into program_settings_map[true];

function set_upgrader_address:
    input r0 as address.private;
    async set_upgrader_address self.caller r0 into r1;
    output r1 as test_shield_swap_multisig_core.aleo/set_upgrader_address.future;

finalize set_upgrader_address:
    input r0 as address.public;
    input r1 as address.public;
    contains program_settings_map[true] into r2;
    assert.eq r2 true;
    get program_settings_map[true] into r3;
    assert.eq r0 r3.upgrader_address;
    assert.eq r3.allow_upgrades true;
    cast r3.allow_upgrades r1 r3.guard_create_wallet into r4 as ProgramSettings;
    set r4 into program_settings_map[true];

function create_wallet:
    input r0 as address.public;
    input r1 as u8.public;
    input r2 as [address; 4u32].private;
    input r3 as [[u8; 20u32]; 4u32].private;
    gt r1 0u8 into r4;
    lte r1 8u8 into r5;
    and r4 r5 into r6;
    assert.eq r6 true;
    cast r0 r2[0u32] into r7 as WalletAleoSigner;
    hash.bhp256 r7 into r8 as field;
    cast r0 r2[1u32] into r9 as WalletAleoSigner;
    hash.bhp256 r9 into r10 as field;
    cast r0 r2[2u32] into r11 as WalletAleoSigner;
    hash.bhp256 r11 into r12 as field;
    cast r0 r2[3u32] into r13 as WalletAleoSigner;
    hash.bhp256 r13 into r14 as field;
    cast r8 r10 r12 r14 into r15 as [field; 4u32];
    cast r0 r3[0u32] into r16 as WalletEcdsaSigner;
    hash.bhp256 r16 into r17 as field;
    cast r0 r3[1u32] into r18 as WalletEcdsaSigner;
    hash.bhp256 r18 into r19 as field;
    cast r0 r3[2u32] into r20 as WalletEcdsaSigner;
    hash.bhp256 r20 into r21 as field;
    cast r0 r3[3u32] into r22 as WalletEcdsaSigner;
    hash.bhp256 r22 into r23 as field;
    cast r17 r19 r21 r23 into r24 as [field; 4u32];
    async create_wallet r0 r1 r2 r3 self.caller r15 r24 into r25;
    output r25 as test_shield_swap_multisig_core.aleo/create_wallet.future;

finalize create_wallet:
    input r0 as address.public;
    input r1 as u8.public;
    input r2 as [address; 4u32].public;
    input r3 as [[u8; 20u32]; 4u32].public;
    input r4 as address.public;
    input r5 as [field; 4u32].public;
    input r6 as [field; 4u32].public;
    contains program_settings_map[true] into r7;
    assert.eq r7 true;
    get program_settings_map[true] into r8;
    is.neq r0 test_shield_swap_multisig_core.aleo into r9;
    and r8.guard_create_wallet r9 into r10;
    branch.eq r10 false to end_then_0_0;
    cast r0 r1 r2 r3 into r11 as GuardedCreateWalletOp;
    hash.bhp256 r11 into r12 as field;
    cast test_shield_swap_multisig_core.aleo r12 into r13 as WalletSigningOpId;
    hash.bhp256 r13 into r14 as field;
    contains completed_signing_ops[r14] into r15;
    assert.eq r15 true;
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    is.eq r0 test_shield_swap_multisig_core.aleo into r16;
    branch.eq r16 false to end_then_1_2;
    assert.eq r4 aleo1sa2gyw566v8k7ezjljs8hg4y6ja3vf0sm0kwzfvw5hvredjq3gys6wvfys;
    branch.eq true true to end_otherwise_1_3;
    position end_then_1_2;
    position end_otherwise_1_3;
    position end_otherwise_0_1;
    contains wallets_map[r0] into r17;
    not r17 into r18;
    assert.eq r18 true;
    cast r0 aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into r19 as WalletAleoSigner;
    hash.bhp256 r19 into r20 as field;
    cast 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 into r21 as [u8; 20u32];
    cast r0 r21 into r22 as WalletEcdsaSigner;
    hash.bhp256 r22 into r23 as field;
    is.neq r5[0u32] r20 into r24;
    branch.eq r24 false to end_then_0_4;
    contains signers_map[r5[0u32]] into r25;
    not r25 into r26;
    assert.eq r26 true;
    set true into signers_map[r5[0u32]];
    branch.eq true true to end_otherwise_0_5;
    position end_then_0_4;
    position end_otherwise_0_5;
    ternary r24 1u8 0u8 into r27;
    is.neq r6[0u32] r23 into r28;
    branch.eq r28 false to end_then_0_6;
    contains signers_map[r6[0u32]] into r29;
    not r29 into r30;
    assert.eq r30 true;
    set true into signers_map[r6[0u32]];
    branch.eq true true to end_otherwise_0_7;
    position end_then_0_6;
    position end_otherwise_0_7;
    ternary r28 1u8 0u8 into r31;
    add r27 r31 into r32;
    is.neq r5[1u32] r20 into r33;
    branch.eq r33 false to end_then_0_8;
    contains signers_map[r5[1u32]] into r34;
    not r34 into r35;
    assert.eq r35 true;
    set true into signers_map[r5[1u32]];
    branch.eq true true to end_otherwise_0_9;
    position end_then_0_8;
    position end_otherwise_0_9;
    ternary r33 1u8 0u8 into r36;
    add r32 r36 into r37;
    is.neq r6[1u32] r23 into r38;
    branch.eq r38 false to end_then_0_10;
    contains signers_map[r6[1u32]] into r39;
    not r39 into r40;
    assert.eq r40 true;
    set true into signers_map[r6[1u32]];
    branch.eq true true to end_otherwise_0_11;
    position end_then_0_10;
    position end_otherwise_0_11;
    ternary r38 1u8 0u8 into r41;
    add r37 r41 into r42;
    is.neq r5[2u32] r20 into r43;
    branch.eq r43 false to end_then_0_12;
    contains signers_map[r5[2u32]] into r44;
    not r44 into r45;
    assert.eq r45 true;
    set true into signers_map[r5[2u32]];
    branch.eq true true to end_otherwise_0_13;
    position end_then_0_12;
    position end_otherwise_0_13;
    ternary r43 1u8 0u8 into r46;
    add r42 r46 into r47;
    is.neq r6[2u32] r23 into r48;
    branch.eq r48 false to end_then_0_14;
    contains signers_map[r6[2u32]] into r49;
    not r49 into r50;
    assert.eq r50 true;
    set true into signers_map[r6[2u32]];
    branch.eq true true to end_otherwise_0_15;
    position end_then_0_14;
    position end_otherwise_0_15;
    ternary r48 1u8 0u8 into r51;
    add r47 r51 into r52;
    is.neq r5[3u32] r20 into r53;
    branch.eq r53 false to end_then_0_16;
    contains signers_map[r5[3u32]] into r54;
    not r54 into r55;
    assert.eq r55 true;
    set true into signers_map[r5[3u32]];
    branch.eq true true to end_otherwise_0_17;
    position end_then_0_16;
    position end_otherwise_0_17;
    ternary r53 1u8 0u8 into r56;
    add r52 r56 into r57;
    is.neq r6[3u32] r23 into r58;
    branch.eq r58 false to end_then_0_18;
    contains signers_map[r6[3u32]] into r59;
    not r59 into r60;
    assert.eq r60 true;
    set true into signers_map[r6[3u32]];
    branch.eq true true to end_otherwise_0_19;
    position end_then_0_18;
    position end_otherwise_0_19;
    ternary r58 1u8 0u8 into r61;
    add r57 r61 into r62;
    gte r62 r1 into r63;
    assert.eq r63 true;
    cast r1 r62 into r64 as WalletSettings;
    set r64 into wallets_map[r0];

function initiate_signing_op:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as u32.public;
    gt r2 0u32 into r3;
    assert.eq r3 true;
    cast r0 r1 into r4 as WalletSigningOpId;
    hash.bhp256 r4 into r5 as field;
    cast r0 self.signer into r6 as WalletAleoSigner;
    hash.bhp256 r6 into r7 as field;
    async initiate_signing_op r0 r5 r7 r2 into r8;
    output r5 as field.private;
    output r8 as test_shield_swap_multisig_core.aleo/initiate_signing_op.future;

finalize initiate_signing_op:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as field.public;
    input r3 as u32.public;
    cast 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 into r4 as [u8; 20u32];
    cast 0u8 0u8 aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc r4 into r5 as AdminOp;
    contains wallets_map[r0] into r6;
    assert.eq r6 true;
    contains completed_signing_ops[r1] into r7;
    not r7 into r8;
    assert.eq r8 true;
    contains pending_signing_ops[r1] into r9;
    not r9 into r10;
    branch.eq r10 false to end_then_0_0;
    cast 0u8 0u32 0u32 0u32 into r11 as WalletSigningOp;
    set r11 into pending_signing_ops[r1];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;
    get pending_signing_ops[r1] into r12;
    lt r12.expires_at_block block.height into r13;
    assert.eq r13 true;
    is.eq r3 4294967295u32 into r14;
    ternary r14 0u32 r3 into r15;
    add block.height r15 into r16;
    ternary r14 4294967295u32 r16 into r17;
    add r12.round 1u32 into r18;
    contains signers_map[r2] into r19;
    branch.eq r19 false to end_then_0_2;
    cast 1u8 block.height r17 r18 into r20 as WalletSigningOp;
    set r20 into pending_signing_ops[r1];
    cast r1 r2 into r21 as SigningOpConfirmation;
    hash.bhp256 r21 into r22 as field;
    get.or_use pending_signing_confirmations[r22] 0u32 into r23;
    gt r18 r23 into r24;
    assert.eq r24 true;
    set r18 into pending_signing_confirmations[r22];
    get wallets_map[r0] into r25;
    gte 1u8 r25.threshold into r26;
    branch.eq r26 false to end_then_1_4;
    set block.height into completed_signing_ops[r1];
    branch.eq true true to end_otherwise_1_5;
    position end_then_1_4;
    position end_otherwise_1_5;
    branch.eq true true to end_otherwise_0_3;
    position end_then_0_2;
    cast 0u8 block.height r17 r18 into r27 as WalletSigningOp;
    set r27 into pending_signing_ops[r1];
    position end_otherwise_0_3;
    contains pending_admin_ops[r1] into r28;
    branch.eq r28 false to end_then_0_6;
    get pending_admin_ops[r1] into r29;
    assert.eq r5 r29;
    branch.eq true true to end_otherwise_0_7;
    position end_then_0_6;
    set r5 into pending_admin_ops[r1];
    position end_otherwise_0_7;

function sign:
    input r0 as address.public;
    input r1 as field.public;
    cast r0 self.signer into r2 as WalletAleoSigner;
    hash.bhp256 r2 into r3 as field;
    async sign r0 r1 r3 into r4;
    output r4 as test_shield_swap_multisig_core.aleo/sign.future;

finalize sign:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as field.public;
    contains wallets_map[r0] into r3;
    assert.eq r3 true;
    contains signers_map[r2] into r4;
    assert.eq r4 true;
    cast r0 r1 into r5 as WalletSigningOpId;
    hash.bhp256 r5 into r6 as field;
    contains pending_signing_ops[r6] into r7;
    assert.eq r7 true;
    get pending_signing_ops[r6] into r8;
    lt r8.expires_at_block block.height into r9;
    not r9 into r10;
    assert.eq r10 true;
    cast r6 r2 into r11 as SigningOpConfirmation;
    hash.bhp256 r11 into r12 as field;
    get.or_use pending_signing_confirmations[r12] 0u32 into r13;
    gt r8.round r13 into r14;
    assert.eq r14 true;
    set r8.round into pending_signing_confirmations[r12];
    add r8.confirmations 1u8 into r15;
    cast r15 r8.initiated_at_block r8.expires_at_block r8.round into r16 as WalletSigningOp;
    set r16 into pending_signing_ops[r6];
    get wallets_map[r0] into r17;
    gte r15 r17.threshold into r18;
    branch.eq r18 false to end_then_0_0;
    set block.height into completed_signing_ops[r6];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;

function sign_for_round:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as u32.public;
    cast r0 self.signer into r3 as WalletAleoSigner;
    hash.bhp256 r3 into r4 as field;
    async sign_for_round r0 r1 r4 r2 into r5;
    output r5 as test_shield_swap_multisig_core.aleo/sign_for_round.future;

finalize sign_for_round:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as field.public;
    input r3 as u32.public;
    contains wallets_map[r0] into r4;
    assert.eq r4 true;
    contains signers_map[r2] into r5;
    assert.eq r5 true;
    cast r0 r1 into r6 as WalletSigningOpId;
    hash.bhp256 r6 into r7 as field;
    contains pending_signing_ops[r7] into r8;
    assert.eq r8 true;
    get pending_signing_ops[r7] into r9;
    lt r9.expires_at_block block.height into r10;
    not r10 into r11;
    assert.eq r11 true;
    is.eq r3 0u32 into r12;
    is.eq r3 r9.round into r13;
    or r12 r13 into r14;
    assert.eq r14 true;
    cast r7 r2 into r15 as SigningOpConfirmation;
    hash.bhp256 r15 into r16 as field;
    get.or_use pending_signing_confirmations[r16] 0u32 into r17;
    gt r9.round r17 into r18;
    assert.eq r18 true;
    set r9.round into pending_signing_confirmations[r16];
    add r9.confirmations 1u8 into r19;
    cast r19 r9.initiated_at_block r9.expires_at_block r9.round into r20 as WalletSigningOp;
    set r20 into pending_signing_ops[r7];
    get wallets_map[r0] into r21;
    gte r19 r21.threshold into r22;
    branch.eq r22 false to end_then_0_0;
    set block.height into completed_signing_ops[r7];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;

function sign_ecdsa:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as [u8; 20u32].public;
    input r3 as [u8; 65u32].public;
    input r4 as u64.public;
    cast r0 r2 into r5 as WalletEcdsaSigner;
    async sign_ecdsa r0 r1 r5 r3 r4 into r6;
    output r6 as test_shield_swap_multisig_core.aleo/sign_ecdsa.future;

finalize sign_ecdsa:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as WalletEcdsaSigner.public;
    input r3 as [u8; 65u32].public;
    input r4 as u64.public;
    contains wallets_map[r0] into r5;
    assert.eq r5 true;
    hash.bhp256 r2 into r6 as field;
    contains signers_map[r6] into r7;
    assert.eq r7 true;
    cast r0 r1 r4 into r8 as WalletSigningOpIdNonce;
    hash.bhp256 r8 into r9 as field;
    contains used_nonces[r9] into r10;
    not r10 into r11;
    assert.eq r11 true;
    set true into used_nonces[r9];
    cast r0 r1 into r12 as WalletSigningOpId;
    hash.bhp256 r12 into r13 as field;
    contains pending_signing_ops[r13] into r14;
    assert.eq r14 true;
    get pending_signing_ops[r13] into r15;
    lt r15.expires_at_block block.height into r16;
    not r16 into r17;
    assert.eq r17 true;
    serialize.bits.raw r9  (field) into r18 ([boolean; 253u32]);
    cast false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false into r19 as [boolean; 256u32];
    cast r18[0u32] r18[1u32] r18[2u32] r18[3u32] r18[4u32] r18[5u32] r18[6u32] r18[7u32] r18[8u32] r18[9u32] r18[10u32] r18[11u32] r18[12u32] r18[13u32] r18[14u32] r18[15u32] r18[16u32] r18[17u32] r18[18u32] r18[19u32] r18[20u32] r18[21u32] r18[22u32] r18[23u32] r18[24u32] r18[25u32] r18[26u32] r18[27u32] r18[28u32] r18[29u32] r18[30u32] r18[31u32] r18[32u32] r18[33u32] r18[34u32] r18[35u32] r18[36u32] r18[37u32] r18[38u32] r18[39u32] r18[40u32] r18[41u32] r18[42u32] r18[43u32] r18[44u32] r18[45u32] r18[46u32] r18[47u32] r18[48u32] r18[49u32] r18[50u32] r18[51u32] r18[52u32] r18[53u32] r18[54u32] r18[55u32] r18[56u32] r18[57u32] r18[58u32] r18[59u32] r18[60u32] r18[61u32] r18[62u32] r18[63u32] r18[64u32] r18[65u32] r18[66u32] r18[67u32] r18[68u32] r18[69u32] r18[70u32] r18[71u32] r18[72u32] r18[73u32] r18[74u32] r18[75u32] r18[76u32] r18[77u32] r18[78u32] r18[79u32] r18[80u32] r18[81u32] r18[82u32] r18[83u32] r18[84u32] r18[85u32] r18[86u32] r18[87u32] r18[88u32] r18[89u32] r18[90u32] r18[91u32] r18[92u32] r18[93u32] r18[94u32] r18[95u32] r18[96u32] r18[97u32] r18[98u32] r18[99u32] r18[100u32] r18[101u32] r18[102u32] r18[103u32] r18[104u32] r18[105u32] r18[106u32] r18[107u32] r18[108u32] r18[109u32] r18[110u32] r18[111u32] r18[112u32] r18[113u32] r18[114u32] r18[115u32] r18[116u32] r18[117u32] r18[118u32] r18[119u32] r18[120u32] r18[121u32] r18[122u32] r18[123u32] r18[124u32] r18[125u32] r18[126u32] r18[127u32] r18[128u32] r18[129u32] r18[130u32] r18[131u32] r18[132u32] r18[133u32] r18[134u32] r18[135u32] r18[136u32] r18[137u32] r18[138u32] r18[139u32] r18[140u32] r18[141u32] r18[142u32] r18[143u32] r18[144u32] r18[145u32] r18[146u32] r18[147u32] r18[148u32] r18[149u32] r18[150u32] r18[151u32] r18[152u32] r18[153u32] r18[154u32] r18[155u32] r18[156u32] r18[157u32] r18[158u32] r18[159u32] r18[160u32] r18[161u32] r18[162u32] r18[163u32] r18[164u32] r18[165u32] r18[166u32] r18[167u32] r18[168u32] r18[169u32] r18[170u32] r18[171u32] r18[172u32] r18[173u32] r18[174u32] r18[175u32] r18[176u32] r18[177u32] r18[178u32] r18[179u32] r18[180u32] r18[181u32] r18[182u32] r18[183u32] r18[184u32] r18[185u32] r18[186u32] r18[187u32] r18[188u32] r18[189u32] r18[190u32] r18[191u32] r18[192u32] r18[193u32] r18[194u32] r18[195u32] r18[196u32] r18[197u32] r18[198u32] r18[199u32] r18[200u32] r18[201u32] r18[202u32] r18[203u32] r18[204u32] r18[205u32] r18[206u32] r18[207u32] r18[208u32] r18[209u32] r18[210u32] r18[211u32] r18[212u32] r18[213u32] r18[214u32] r18[215u32] r18[216u32] r18[217u32] r18[218u32] r18[219u32] r18[220u32] r18[221u32] r18[222u32] r18[223u32] r18[224u32] r18[225u32] r18[226u32] r18[227u32] r18[228u32] r18[229u32] r18[230u32] r18[231u32] r18[232u32] r18[233u32] r18[234u32] r18[235u32] r18[236u32] r18[237u32] r18[238u32] r18[239u32] r18[240u32] r18[241u32] r18[242u32] r18[243u32] r18[244u32] r18[245u32] r18[246u32] r18[247u32] r18[248u32] r18[249u32] r18[250u32] r18[251u32] r18[252u32] r19[253u32] r19[254u32] r19[255u32] into r20 as [boolean; 256u32];
    deserialize.bits.raw r20  ([boolean; 256u32]) into r21 ([u8; 32u32]);
    cast 25u8 69u8 116u8 104u8 101u8 114u8 101u8 117u8 109u8 32u8 83u8 105u8 103u8 110u8 101u8 100u8 32u8 77u8 101u8 115u8 115u8 97u8 103u8 101u8 58u8 10u8 51u8 50u8 r21[0u32] r21[1u32] r21[2u32] r21[3u32] r21[4u32] r21[5u32] r21[6u32] r21[7u32] r21[8u32] r21[9u32] r21[10u32] r21[11u32] r21[12u32] r21[13u32] r21[14u32] r21[15u32] r21[16u32] r21[17u32] r21[18u32] r21[19u32] r21[20u32] r21[21u32] r21[22u32] r21[23u32] r21[24u32] r21[25u32] r21[26u32] r21[27u32] r21[28u32] r21[29u32] r21[30u32] r21[31u32] into r22 as [u8; 60u32];
    hash.keccak256.native.raw r22 into r23 as [boolean; 256u32];
    deserialize.bits.raw r23  ([boolean; 256u32]) into r24 ([u8; 32u32]);
    ecdsa.verify.digest.eth r3 r2.ecdsa_signer r24 into r25;
    assert.eq r25 true;
    cast r13 r6 into r26 as SigningOpConfirmation;
    hash.bhp256 r26 into r27 as field;
    get.or_use pending_signing_confirmations[r27] 0u32 into r28;
    gt r15.round r28 into r29;
    assert.eq r29 true;
    set r15.round into pending_signing_confirmations[r27];
    add r15.confirmations 1u8 into r30;
    cast r30 r15.initiated_at_block r15.expires_at_block r15.round into r31 as WalletSigningOp;
    set r31 into pending_signing_ops[r13];
    get wallets_map[r0] into r32;
    gte r30 r32.threshold into r33;
    branch.eq r33 false to end_then_0_0;
    set block.height into completed_signing_ops[r13];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;

function sign_ecdsa_for_round:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as [u8; 20u32].public;
    input r3 as [u8; 65u32].public;
    input r4 as u64.public;
    input r5 as u32.public;
    cast r0 r2 into r6 as WalletEcdsaSigner;
    async sign_ecdsa_for_round r0 r1 r6 r3 r4 r5 into r7;
    output r7 as test_shield_swap_multisig_core.aleo/sign_ecdsa_for_round.future;

finalize sign_ecdsa_for_round:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as WalletEcdsaSigner.public;
    input r3 as [u8; 65u32].public;
    input r4 as u64.public;
    input r5 as u32.public;
    contains wallets_map[r0] into r6;
    assert.eq r6 true;
    hash.bhp256 r2 into r7 as field;
    contains signers_map[r7] into r8;
    assert.eq r8 true;
    cast r0 r1 r4 into r9 as WalletSigningOpIdNonce;
    hash.bhp256 r9 into r10 as field;
    contains used_nonces[r10] into r11;
    not r11 into r12;
    assert.eq r12 true;
    set true into used_nonces[r10];
    cast r0 r1 into r13 as WalletSigningOpId;
    hash.bhp256 r13 into r14 as field;
    contains pending_signing_ops[r14] into r15;
    assert.eq r15 true;
    get pending_signing_ops[r14] into r16;
    lt r16.expires_at_block block.height into r17;
    not r17 into r18;
    assert.eq r18 true;
    is.eq r5 0u32 into r19;
    is.eq r5 r16.round into r20;
    or r19 r20 into r21;
    assert.eq r21 true;
    serialize.bits.raw r10  (field) into r22 ([boolean; 253u32]);
    cast false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false false into r23 as [boolean; 256u32];
    cast r22[0u32] r22[1u32] r22[2u32] r22[3u32] r22[4u32] r22[5u32] r22[6u32] r22[7u32] r22[8u32] r22[9u32] r22[10u32] r22[11u32] r22[12u32] r22[13u32] r22[14u32] r22[15u32] r22[16u32] r22[17u32] r22[18u32] r22[19u32] r22[20u32] r22[21u32] r22[22u32] r22[23u32] r22[24u32] r22[25u32] r22[26u32] r22[27u32] r22[28u32] r22[29u32] r22[30u32] r22[31u32] r22[32u32] r22[33u32] r22[34u32] r22[35u32] r22[36u32] r22[37u32] r22[38u32] r22[39u32] r22[40u32] r22[41u32] r22[42u32] r22[43u32] r22[44u32] r22[45u32] r22[46u32] r22[47u32] r22[48u32] r22[49u32] r22[50u32] r22[51u32] r22[52u32] r22[53u32] r22[54u32] r22[55u32] r22[56u32] r22[57u32] r22[58u32] r22[59u32] r22[60u32] r22[61u32] r22[62u32] r22[63u32] r22[64u32] r22[65u32] r22[66u32] r22[67u32] r22[68u32] r22[69u32] r22[70u32] r22[71u32] r22[72u32] r22[73u32] r22[74u32] r22[75u32] r22[76u32] r22[77u32] r22[78u32] r22[79u32] r22[80u32] r22[81u32] r22[82u32] r22[83u32] r22[84u32] r22[85u32] r22[86u32] r22[87u32] r22[88u32] r22[89u32] r22[90u32] r22[91u32] r22[92u32] r22[93u32] r22[94u32] r22[95u32] r22[96u32] r22[97u32] r22[98u32] r22[99u32] r22[100u32] r22[101u32] r22[102u32] r22[103u32] r22[104u32] r22[105u32] r22[106u32] r22[107u32] r22[108u32] r22[109u32] r22[110u32] r22[111u32] r22[112u32] r22[113u32] r22[114u32] r22[115u32] r22[116u32] r22[117u32] r22[118u32] r22[119u32] r22[120u32] r22[121u32] r22[122u32] r22[123u32] r22[124u32] r22[125u32] r22[126u32] r22[127u32] r22[128u32] r22[129u32] r22[130u32] r22[131u32] r22[132u32] r22[133u32] r22[134u32] r22[135u32] r22[136u32] r22[137u32] r22[138u32] r22[139u32] r22[140u32] r22[141u32] r22[142u32] r22[143u32] r22[144u32] r22[145u32] r22[146u32] r22[147u32] r22[148u32] r22[149u32] r22[150u32] r22[151u32] r22[152u32] r22[153u32] r22[154u32] r22[155u32] r22[156u32] r22[157u32] r22[158u32] r22[159u32] r22[160u32] r22[161u32] r22[162u32] r22[163u32] r22[164u32] r22[165u32] r22[166u32] r22[167u32] r22[168u32] r22[169u32] r22[170u32] r22[171u32] r22[172u32] r22[173u32] r22[174u32] r22[175u32] r22[176u32] r22[177u32] r22[178u32] r22[179u32] r22[180u32] r22[181u32] r22[182u32] r22[183u32] r22[184u32] r22[185u32] r22[186u32] r22[187u32] r22[188u32] r22[189u32] r22[190u32] r22[191u32] r22[192u32] r22[193u32] r22[194u32] r22[195u32] r22[196u32] r22[197u32] r22[198u32] r22[199u32] r22[200u32] r22[201u32] r22[202u32] r22[203u32] r22[204u32] r22[205u32] r22[206u32] r22[207u32] r22[208u32] r22[209u32] r22[210u32] r22[211u32] r22[212u32] r22[213u32] r22[214u32] r22[215u32] r22[216u32] r22[217u32] r22[218u32] r22[219u32] r22[220u32] r22[221u32] r22[222u32] r22[223u32] r22[224u32] r22[225u32] r22[226u32] r22[227u32] r22[228u32] r22[229u32] r22[230u32] r22[231u32] r22[232u32] r22[233u32] r22[234u32] r22[235u32] r22[236u32] r22[237u32] r22[238u32] r22[239u32] r22[240u32] r22[241u32] r22[242u32] r22[243u32] r22[244u32] r22[245u32] r22[246u32] r22[247u32] r22[248u32] r22[249u32] r22[250u32] r22[251u32] r22[252u32] r23[253u32] r23[254u32] r23[255u32] into r24 as [boolean; 256u32];
    deserialize.bits.raw r24  ([boolean; 256u32]) into r25 ([u8; 32u32]);
    cast 25u8 69u8 116u8 104u8 101u8 114u8 101u8 117u8 109u8 32u8 83u8 105u8 103u8 110u8 101u8 100u8 32u8 77u8 101u8 115u8 115u8 97u8 103u8 101u8 58u8 10u8 51u8 50u8 r25[0u32] r25[1u32] r25[2u32] r25[3u32] r25[4u32] r25[5u32] r25[6u32] r25[7u32] r25[8u32] r25[9u32] r25[10u32] r25[11u32] r25[12u32] r25[13u32] r25[14u32] r25[15u32] r25[16u32] r25[17u32] r25[18u32] r25[19u32] r25[20u32] r25[21u32] r25[22u32] r25[23u32] r25[24u32] r25[25u32] r25[26u32] r25[27u32] r25[28u32] r25[29u32] r25[30u32] r25[31u32] into r26 as [u8; 60u32];
    hash.keccak256.native.raw r26 into r27 as [boolean; 256u32];
    deserialize.bits.raw r27  ([boolean; 256u32]) into r28 ([u8; 32u32]);
    ecdsa.verify.digest.eth r3 r2.ecdsa_signer r28 into r29;
    assert.eq r29 true;
    cast r14 r7 into r30 as SigningOpConfirmation;
    hash.bhp256 r30 into r31 as field;
    get.or_use pending_signing_confirmations[r31] 0u32 into r32;
    gt r16.round r32 into r33;
    assert.eq r33 true;
    set r16.round into pending_signing_confirmations[r31];
    add r16.confirmations 1u8 into r34;
    cast r34 r16.initiated_at_block r16.expires_at_block r16.round into r35 as WalletSigningOp;
    set r35 into pending_signing_ops[r14];
    get wallets_map[r0] into r36;
    gte r34 r36.threshold into r37;
    branch.eq r37 false to end_then_0_0;
    set block.height into completed_signing_ops[r14];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;

closure assert_valid_admin_op:
    input r0 as AdminOp;
    is.eq r0.op 1u8 into r1;
    not r1 into r2;
    is.neq r0.threshold 0u8 into r3;
    or r3 r2 into r4;
    assert.eq r4 true;
    is.eq r0.aleo_signer aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into r5;
    or r5 r2 into r6;
    assert.eq r6 true;
    cast 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 into r7 as [u8; 20u32];
    is.eq r0.ecdsa_signer r7 into r8;
    or r8 r2 into r9;
    assert.eq r9 true;
    is.eq r0.op 2u8 into r10;
    is.eq r0.op 3u8 into r11;
    or r10 r11 into r12;
    and r2 r12 into r13;
    not r13 into r14;
    is.eq r0.threshold 0u8 into r15;
    or r15 r14 into r16;
    assert.eq r16 true;
    is.eq r0.aleo_signer aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into r17;
    is.eq r0.ecdsa_signer r7 into r18;
    and r17 r18 into r19;
    and r13 r19 into r20;
    not r20 into r21;
    assert.eq r21 true;
    not r19 into r22;
    is.neq r0.aleo_signer aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into r23;
    is.neq r0.ecdsa_signer r7 into r24;
    and r23 r24 into r25;
    and r13 r22 into r26;
    and r26 r25 into r27;
    not r27 into r28;
    assert.eq r28 true;
    not r12 into r29;
    and r2 r29 into r30;
    not r30 into r31;
    assert.eq r31 true;

function init_admin_op:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as u32.public;
    input r3 as AdminOp.private;
    call assert_valid_admin_op r3;
    cast r0 r1 into r4 as WalletSigningOpId;
    hash.bhp256 r4 into r5 as field;
    cast r0 self.signer into r6 as WalletAleoSigner;
    hash.bhp256 r6 into r7 as field;
    async init_admin_op r0 r5 r7 r2 r3 into r8;
    output r5 as field.private;
    output r8 as test_shield_swap_multisig_core.aleo/init_admin_op.future;

finalize init_admin_op:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as field.public;
    input r3 as u32.public;
    input r4 as AdminOp.public;
    contains wallets_map[r0] into r5;
    assert.eq r5 true;
    contains completed_signing_ops[r1] into r6;
    not r6 into r7;
    assert.eq r7 true;
    contains pending_signing_ops[r1] into r8;
    not r8 into r9;
    branch.eq r9 false to end_then_0_0;
    cast 0u8 0u32 0u32 0u32 into r10 as WalletSigningOp;
    set r10 into pending_signing_ops[r1];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;
    get pending_signing_ops[r1] into r11;
    lt r11.expires_at_block block.height into r12;
    assert.eq r12 true;
    is.eq r3 4294967295u32 into r13;
    ternary r13 0u32 r3 into r14;
    add block.height r14 into r15;
    ternary r13 4294967295u32 r15 into r16;
    add r11.round 1u32 into r17;
    contains signers_map[r2] into r18;
    branch.eq r18 false to end_then_0_2;
    cast 1u8 block.height r16 r17 into r19 as WalletSigningOp;
    set r19 into pending_signing_ops[r1];
    cast r1 r2 into r20 as SigningOpConfirmation;
    hash.bhp256 r20 into r21 as field;
    get.or_use pending_signing_confirmations[r21] 0u32 into r22;
    gt r17 r22 into r23;
    assert.eq r23 true;
    set r17 into pending_signing_confirmations[r21];
    get wallets_map[r0] into r24;
    gte 1u8 r24.threshold into r25;
    branch.eq r25 false to end_then_1_4;
    set block.height into completed_signing_ops[r1];
    branch.eq true true to end_otherwise_1_5;
    position end_then_1_4;
    position end_otherwise_1_5;
    branch.eq true true to end_otherwise_0_3;
    position end_then_0_2;
    cast 0u8 block.height r16 r17 into r26 as WalletSigningOp;
    set r26 into pending_signing_ops[r1];
    position end_otherwise_0_3;
    contains pending_admin_ops[r1] into r27;
    branch.eq r27 false to end_then_0_6;
    get pending_admin_ops[r1] into r28;
    assert.eq r4 r28;
    branch.eq true true to end_otherwise_0_7;
    position end_then_0_6;
    set r4 into pending_admin_ops[r1];
    position end_otherwise_0_7;

function exec_admin_op:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as AdminOp.private;
    call assert_valid_admin_op r2;
    cast r0 r1 into r3 as WalletSigningOpId;
    hash.bhp256 r3 into r4 as field;
    async exec_admin_op r0 r4 r2 into r5;
    output r5 as test_shield_swap_multisig_core.aleo/exec_admin_op.future;

finalize exec_admin_op:
    input r0 as address.public;
    input r1 as field.public;
    input r2 as AdminOp.public;
    contains wallets_map[r0] into r3;
    assert.eq r3 true;
    contains completed_signing_ops[r1] into r4;
    assert.eq r4 true;
    get pending_admin_ops[r1] into r5;
    assert.eq r5 r2;
    remove pending_admin_ops[r1];
    get wallets_map[r0] into r6;
    is.eq r2.op 1u8 into r7;
    branch.eq r7 false to end_then_0_0;
    lte r2.threshold r6.num_signers into r8;
    assert.eq r8 true;
    cast r2.threshold r6.num_signers into r9 as WalletSettings;
    set r9 into wallets_map[r0];
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    is.eq r2.op 2u8 into r10;
    branch.eq r10 false to end_then_1_2;
    is.eq r2.op 2u8 into r11;
    is.eq r2.op 3u8 into r12;
    or r11 r12 into r13;
    assert.eq r13 true;
    is.neq r2.aleo_signer aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into r14;
    cast r0 r2.aleo_signer into r15 as WalletAleoSigner;
    hash.bhp256 r15 into r16 as field;
    ternary r14 r16 0field into r17;
    cast 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 into r18 as [u8; 20u32];
    is.neq r2.ecdsa_signer r18 into r19;
    not r19 into r20;
    is.eq r17 0field into r21;
    or r21 r20 into r22;
    assert.eq r22 true;
    cast r0 r2.ecdsa_signer into r23 as WalletEcdsaSigner;
    hash.bhp256 r23 into r24 as field;
    ternary r19 r24 r17 into r25;
    assert.neq r25 0field;
    contains signers_map[r25] into r26;
    not r26 into r27;
    assert.eq r27 true;
    set true into signers_map[r25];
    add r6.num_signers 1u8 into r28;
    cast r6.threshold r28 into r29 as WalletSettings;
    set r29 into wallets_map[r0];
    branch.eq true true to end_otherwise_1_3;
    position end_then_1_2;
    is.eq r2.op 3u8 into r30;
    branch.eq r30 false to end_then_2_4;
    is.eq r2.op 2u8 into r31;
    is.eq r2.op 3u8 into r32;
    or r31 r32 into r33;
    assert.eq r33 true;
    is.neq r2.aleo_signer aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc into r34;
    cast r0 r2.aleo_signer into r35 as WalletAleoSigner;
    hash.bhp256 r35 into r36 as field;
    ternary r34 r36 0field into r37;
    cast 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 0u8 into r38 as [u8; 20u32];
    is.neq r2.ecdsa_signer r38 into r39;
    not r39 into r40;
    is.eq r37 0field into r41;
    or r41 r40 into r42;
    assert.eq r42 true;
    cast r0 r2.ecdsa_signer into r43 as WalletEcdsaSigner;
    hash.bhp256 r43 into r44 as field;
    ternary r39 r44 r37 into r45;
    assert.neq r45 0field;
    contains signers_map[r45] into r46;
    assert.eq r46 true;
    remove signers_map[r45];
    sub r6.num_signers 1u8 into r47;
    gte r47 r6.threshold into r48;
    assert.eq r48 true;
    sub r6.num_signers 1u8 into r49;
    cast r6.threshold r49 into r50 as WalletSettings;
    set r50 into wallets_map[r0];
    branch.eq true true to end_otherwise_2_5;
    position end_then_2_4;
    assert.eq false true;
    position end_otherwise_2_5;
    position end_otherwise_1_3;
    position end_otherwise_0_1;

function get_signing_op_id_hash:
    input r0 as address.public;
    input r1 as field.public;
    cast r0 r1 into r2 as WalletSigningOpId;
    hash.bhp256 r2 into r3 as field;
    output r3 as field.private;

function guarded_create_wallet_op_hash:
    input r0 as address.public;
    input r1 as u8.public;
    input r2 as [address; 4u32].private;
    input r3 as [[u8; 20u32]; 4u32].private;
    cast r0 r1 r2 r3 into r4 as GuardedCreateWalletOp;
    hash.bhp256 r4 into r5 as field;
    output r5 as field.private;

function assert_signing_completed:
    input r0 as address.public;
    input r1 as field.public;
    async assert_signing_completed r0 r1 into r2;
    output r2 as test_shield_swap_multisig_core.aleo/assert_signing_completed.future;

finalize assert_signing_completed:
    input r0 as address.public;
    input r1 as field.public;
    cast r0 r1 into r2 as WalletSigningOpId;
    hash.bhp256 r2 into r3 as field;
    contains completed_signing_ops[r3] into r4;
    assert.eq r4 true;

constructor:
    gt edition 0u16 into r0;
    branch.eq r0 false to end_then_0_0;
    contains program_settings_map[true] into r1;
    assert.eq r1 true;
    get program_settings_map[true] into r2;
    assert.eq r2.allow_upgrades true;
    assert.eq program_owner r2.upgrader_address;
    branch.eq true true to end_otherwise_0_1;
    position end_then_0_0;
    position end_otherwise_0_1;

