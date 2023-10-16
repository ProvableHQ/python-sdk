program binary_tree_pahts.aleo;



function main:
    input r0 as boolean.public;
    input r1 as boolean.private;
    input r2 as boolean.private;
    input r3 as boolean.private;
    input r4 as boolean.private;
    input r5 as boolean.private;
    input r6 as boolean.private;
    input r7 as boolean.private;
    input r8 as boolean.private;
    input r9 as boolean.private;
    cast 0i16 into r10 as field;
    and r0 r1 into r11;
    and r11 r2 into r12;
    and r12 r3 into r13;
    and r13 r4 into r14;
    and r14 r5 into r15;
    and r15 r6 into r16;
    and r16 r7 into r17;
    and r17 r8 into r18;
    not r9 into r19;
    and r18 r19 into r20;
    cast r20 into r21 as field;
    add r10 r21 into r22;
    output r20 as boolean.private;
