import json
import struct

def bf16_to_f32(val_u16):
    # BF16 is just the upper 16 bits of an IEEE 754 float32
    return struct.unpack('<f', struct.pack('<I', val_u16 << 16))[0]

def fp16_to_f32(val_u16):
    # IEEE 754 half-precision to float32
    # sign: 1, exponent: 5, fraction: 10
    sign = (val_u16 >> 15) & 1
    exp = (val_u16 >> 10) & 0x1F
    frac = val_u16 & 0x3FF
    if exp == 0:
        if frac == 0:
            return -0.0 if sign else 0.0
        else:
            # subnormal
            return (-1)**sign * (2**-14) * (frac / 1024.0)
    elif exp == 0x1F:
        if frac == 0:
            return float('-inf') if sign else float('inf')
        else:
            return float('nan')
    else:
        return (-1)**sign * (2**(exp - 15)) * (1.0 + frac / 1024.0)

filepath = '/local/models/google/gemma-4-12B-it/model.safetensors'
with open(filepath, 'rb') as f:
    header_size_bytes = f.read(8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]
    header_json_bytes = f.read(header_size)
    header = json.loads(header_json_bytes.decode('utf-8'))
    
    data_start = 8 + header_size
    
    for tensor_name in ["model.language_model.norm.weight", "model.language_model.layers.0.input_layernorm.weight"]:
        if tensor_name in header:
            meta = header[tensor_name]
            print(f"\nTensor: {tensor_name}")
            print(f"  Metadata: {meta}")
            offsets = meta["data_offsets"]
            f.seek(data_start + offsets[0])
            raw_data = f.read(min(20, offsets[1] - offsets[0]))
            
            u16_vals = []
            for i in range(0, len(raw_data), 2):
                val_u16 = struct.unpack('<H', raw_data[i:i+2])[0]
                u16_vals.append(val_u16)
            
            print(f"  Raw Hex: {raw_data.hex()}")
            print(f"  U16 values: {u16_vals}")
            
            bf16_decoded = [bf16_to_f32(v) for v in u16_vals]
            fp16_decoded = [fp16_to_f32(v) for v in u16_vals]
            print(f"  Decoded as BF16: {bf16_decoded}")
            print(f"  Decoded as FP16: {fp16_decoded}")
        else:
            print(f"\nTensor {tensor_name} not found in header!")
