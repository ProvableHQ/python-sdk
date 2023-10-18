// The 'binary_tree_pahts' program.
program binary_tree_pahts.aleo {
    transition main(public a: bool, b: bool, c: bool, d: bool, e: bool, f: bool, g: bool, h: bool, i: bool, j: bool) -> bool {
        let counter_a: field = (0i16 as field);
        let res: bool = a & b & c & d & e & f & g & h & i & (! j);
        counter_a = counter_a + res as field;
        return res;
    }
}
