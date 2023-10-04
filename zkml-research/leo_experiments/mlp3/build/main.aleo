program mlp3.aleo;



function main:
    input r0 as i32.public;
    cast 0i32 into r1 as field;
    cast 50i32 into r2 as field;
    add r1 r2 into r3;
    cast 50i32 into r4 as field;
    add r3 r4 into r5;
    cast r5 into r6 as i32;
    output r6 as i32.private;
    output r0 as i32.private;
