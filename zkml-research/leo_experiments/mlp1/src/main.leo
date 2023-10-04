// The 'mlp1' program.
program mlp1.aleo {
    transition main(public a: i32, b: i32) -> (i128, i128) {
        let c: i32 = a;
        let c_field: field = c as field;


        //let bias : field = 50field;
        //let c_field_relu: field = relu(c_field) + bias;

        let bias : i32 = -50i32;
        let c_field_relu: field = relu(c_field) + bias as field;


        let d_field: field = b as field;
        d_field = d_field / c_field;
        let d_field_relu: field = relu(d_field);

        let c_output: i32 = c_field_relu as i32;
        let output_1_field : field = (-90i128 as field);

        // this fails
        //let output_0_field : field = (-90i128 as field)*(118i128 as field)/(128i128 as field);

        let field_1 : field = -90i128 as field;
        let field_2 : field = 118i128 as field;
        let field_3 : field = 128i128 as field;
        //let output_0_field : field = field_1 * field_2;

        // this works, but fails if we divide by more than 90
        let output_0_field : field = (118i128 as field)*(90i128 as field);
        output_0_field = output_0_field / (90i128 as field);

        return (output_0_field as i128, output_1_field as i128);
    }

    function relu(x: field) -> field {
        let x_integer: i32 = x as i32; // around 33 constraints
        if x_integer < 0i32 {
            return 0field;
        } else {
            return x_integer as field;
        }
    }
}
