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
    mul r6 r2 into r7;
    mul r7 r2 into r8;
    call relu r8 into r9;
    cast r5 into r10 as i32;
    cast r9 into r11 as i32;
    output r10 as i32.private;
    output r11 as i32.private;
