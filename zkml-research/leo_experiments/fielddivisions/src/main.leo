// The 'fielddivisions' program.
program fielddivisions.aleo {
    transition main(public a: u32, b: u32) -> u32 {
        let c: u32 = a + b;
        let output_0_field : field = (118u32 as field)*(90u32 as field);
        
        // The first line works. Once we divide by something larger than 90, it breaks. So commenting out the first line and uncommenting the second line breaks it.
        output_0_field = output_0_field / (90u32 as field);
        //output_0_field = output_0_field / (91u32 as field);

        return (output_0_field as u32);
    }
}
