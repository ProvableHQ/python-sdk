// The 'mlp1' program.
program mlp1.aleo {
    transition main(public a: i32, b: i32) -> (i32, i32) {
        let c: i32 = a;
        let c_field: field = c as field;


        //let bias : field = 50field;
        //let c_field_relu: field = relu(c_field) + bias;

        let bias : i32 = -50i32;
        let c_field_relu: field = relu(c_field) + bias as field;


        let d_field: field = b as field;
        d_field = d_field * c_field * c_field;
        let d_field_relu: field = relu(d_field);

        let c_output: i32 = c_field_relu as i32;
        return (c_output, d_field_relu as i32);
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
