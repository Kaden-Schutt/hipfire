import sys
import struct
import json

def fp16_to_f32(val_u16):
    sign = (val_u16 >> 15) & 1
    exp = (val_u16 >> 10) & 0x1F
    frac = val_u16 & 0x3FF
    if exp == 0:
        if frac == 0:
            return -0.0 if sign else 0.0
        else:
            return (-1)**sign * (2**-14) * (frac / 1024.0)
    elif exp == 0x1F:
        if frac == 0:
            return float('-inf') if sign else float('inf')
        else:
            return float('nan')
    else:
        return (-1)**sign * (2**(exp - 15)) * (1.0 + frac / 1024.0)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_hfq_norms.py <path_to_hfq>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    with open(filepath, 'rb') as f:
        header = f.read(32)
        magic, version, arch_id, n_tensors, metadata_offset, data_offset = struct.unpack('<4sIIiQQ', header)
        if magic != b'HFQM':
            print("Not an HFQ file!")
            sys.exit(1)
            
        print(f"Arch ID: {arch_id}, Tensors: {n_tensors}")
        
        # Read metadata JSON
        f.seek(metadata_offset)
        meta_data = f.read(data_offset - metadata_offset)
        # Find JSON end by scanning brace depth
        brace_depth = 0
        in_string = False
        escape = False
        json_end = 0
        for i, b in enumerate(meta_data):
            if escape:
                escape = False
                continue
            if b == ord('\\') and in_string:
                escape = True
                continue
            if b == ord('"'):
                in_string = not in_string
                continue
            if not in_string:
                if b == ord('{'):
                    brace_depth += 1
                elif b == ord('}'):
                    brace_depth -= 1
                    if brace_depth == 0:
                        json_end = i + 1
                        break
        
        # Parse tensor index
        pos = metadata_offset + json_end
        f.seek(pos)
        idx_n = struct.unpack('<I', f.read(4))[0]
        assert idx_n == n_tensors
        
        tensors = []
        cumulative_offset = data_offset
        for _ in range(n_tensors):
            name_len = struct.unpack('<H', f.read(2))[0]
            name = f.read(name_len).decode('utf-8')
            quant_type = struct.unpack('<B', f.read(1))[0]
            n_dims = struct.unpack('<B', f.read(1))[0]
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack('<I', f.read(4))[0])
            group_size = struct.unpack('<I', f.read(4))[0]
            data_size = struct.unpack('<Q', f.read(8))[0]
            
            tensors.append({
                'name': name,
                'quant_type': quant_type,
                'shape': shape,
                'group_size': group_size,
                'offset': cumulative_offset,
                'size': data_size
            })
            cumulative_offset += data_size
            
        # Inspect norms
        count = 0
        for t in tensors:
            if "norm" in t['name'] or "scalar" in t['name']:
                count += 1
                if count > 10:
                    print("... (more norm tensors omitted)")
                    break
                f.seek(t['offset'])
                raw = f.read(min(20, t['size']))
                u16_vals = []
                for idx in range(0, len(raw), 2):
                    u16_vals.append(struct.unpack('<H', raw[idx:idx+2])[0])
                
                # Check if it's FP16 or FP32
                if t['quant_type'] == 1: # FP16
                    decoded = [fp16_to_f32(v) for v in u16_vals]
                elif t['quant_type'] == 2: # FP32
                    # decode as float32
                    decoded = []
                    for idx in range(0, len(raw), 4):
                        if idx + 4 <= len(raw):
                            decoded.append(struct.unpack('<f', raw[idx:idx+4])[0])
                else:
                    decoded = f"quant_type={t['quant_type']}"
                    
                print(f"Tensor: {t['name']}")
                print(f"  Shape: {t['shape']}, QuantType: {t['quant_type']}")
                print(f"  First values: {decoded}")

if __name__ == '__main__':
    main()
