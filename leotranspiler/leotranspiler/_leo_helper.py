def _get_leo_type(signed, value_bits):
    if(signed):
        if(value_bits <= 7):
            return "i8"
        elif(value_bits <= 15):
            return "i16"
        elif(value_bits <= 31):
            return "i32"
        elif(value_bits <= 63):
            return "i64"
        elif(value_bits <= 127):
            return "i128"
        else:
            raise ValueError("No leo type for signed value with more than 127 bits. Try quantizing the model and/or the data.")
    else:
        if(value_bits <= 8):
            return "u8"
        elif(value_bits <= 16):
            return "u16"
        elif(value_bits <= 32):
            return "u32"
        elif(value_bits <= 64):
            return "u64"
        elif(value_bits <= 128):
            return "u128"
        else:
            raise ValueError("No leo type for unsigned value with more than 128 bits. Try quantizing the model and/or the data.")