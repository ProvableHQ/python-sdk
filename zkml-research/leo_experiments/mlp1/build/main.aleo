program mlp1.aleo;



closure relu:
    input r0 as field;
    cast r0 into r1 as i32;
    lt r1 0i32 into r2;
    cast r1 into r3 as field;
    ternary r2 0field r3 into r4;
    output r4 as field;


function main:
    input r0 as i32.public;
    input r1 as i32.private;
    cast r0 into r2 as field;
    call relu r2 into r3;
    cast -50i32 into r4 as field;
    add r3 r4 into r5;
    cast r1 into r6 as field;
    div r6 r2 into r7;
    call relu r7 into r8;
    cast r5 into r9 as i32;
    cast -90i128 into r10 as field;
    cast -90i128 into r11 as field;
    cast 118i128 into r12 as field;
    cast 128i128 into r13 as field;
    cast 118i128 into r14 as field;
    cast 90i128 into r15 as field;
    mul r14 r15 into r16;
    cast 91i128 into r17 as field;
    div r16 r17 into r18;
    cast r18 into r19 as i128;
    cast r10 into r20 as i128;
    output r19 as i128.private;
    output r20 as i128.private;
