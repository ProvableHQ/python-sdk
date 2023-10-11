program fielddivisions.aleo;



function main:
    input r0 as u32.public;
    input r1 as u32.private;
    add r0 r1 into r2;
    cast 118u32 into r3 as field;
    cast 90u32 into r4 as field;
    mul r3 r4 into r5;
    cast 90u32 into r6 as field;
    div r5 r6 into r7;
    cast r7 into r8 as u32;
    output r8 as u32.private;
