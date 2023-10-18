program mlp3.aleo {
    transition main(public a: i32) -> (i32, i32) {
        let a_field: field = 0i32 as field;

        let bias : i32 = 50i32;
        let bias_2 : i32 = 50i32;

        let field_sum: field = a_field + bias as field + bias_2 as field;
        let output: i32 = field_sum as i32;

        return (output, a);
    }

}
