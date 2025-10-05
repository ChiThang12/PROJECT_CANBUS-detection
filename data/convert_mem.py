import pandas as pd
import numpy as np

# Đọc file CSV
df = pd.read_csv('tree.csv')

# Parse hex features sang integer
df['Feature'] = df['Feature'].apply(lambda x: int(str(x), 16) if pd.notna(x) and x != -1 else -1)

print("="*80)
print("ORIGINAL DATA:")
print("="*80)
print(df.head(10))

# ============= QUANTIZE THRESHOLD (PER-FEATURE) =============
def quantize_threshold_per_feature(df, bits=32):
    """
    Lượng tử hóa threshold theo từng feature riêng biệt
    Điều này quan trọng vì mỗi feature có range khác nhau
    """
    df['Threshold_Quant'] = 0
    
    # Thống kê threshold theo từng feature
    feature_stats = {}
    unique_features = df[df['Feature'] != -1]['Feature'].unique()
    
    print(f"\nThreshold Statistics per Feature:")
    print("="*80)
    
    for feat in unique_features:
        feature_mask = (df['Feature'] == feat) & (df['Threshold'] != 0)
        thresholds = df.loc[feature_mask, 'Threshold'].values
        
        if len(thresholds) > 0:
            min_val = np.min(thresholds)
            max_val = np.max(thresholds)
            
            feature_stats[feat] = {
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val,
                'count': len(thresholds)
            }
            
            print(f"Feature 0x{feat:02X}: min={min_val:.2e}, max={max_val:.2e}, "
                  f"range={max_val - min_val:.2e}, count={len(thresholds)}")
            
            # Quantize thresholds for this feature
            if max_val > min_val:
                scale = ((2 ** bits) - 1) / (max_val - min_val)
                for idx in df[feature_mask].index:
                    thresh = df.loc[idx, 'Threshold']
                    # Normalize to [0, 2^bits - 1]
                    quantized = int((thresh - min_val) * scale)
                    df.loc[idx, 'Threshold_Quant'] = quantized
            else:
                # All thresholds same value
                df.loc[feature_mask, 'Threshold_Quant'] = (2 ** (bits - 1))
    
    print("="*80)
    
    # Convert to Q32.32 format (integer part only, fractional = 0)
    # Shift quantized value to integer part [63:32]
    df['Threshold_Q32'] = df['Threshold_Quant'].apply(lambda x: x << 32)
    
    return df, feature_stats

df, feature_stats = quantize_threshold_per_feature(df, bits=32)

# ============= CONVERT TO 95-BIT BINARY (khớp với RTL) =============
def to_binary_95bit(row):
    """
    Convert sang 95-bit format theo tree_memory.v:
    [94:86] = Node ID (9 bits)
    [85:84] = Feature Index (2 bits) - chỉ hỗ trợ 3 features (00, 01, 10)
    [83:20] = Threshold (64 bits) - Q32.32 format
    [19:11] = Left Child (9 bits)
    [10:2]  = Right Child (9 bits)
    [1:0]   = Prediction (2 bits)
    """
    
    # Node ID (9 bits)
    node = int(row['Node'])
    if node >= 512:  # Max 9-bit = 511
        print(f"WARNING: Node {node} exceeds 9-bit range!")
        node = 511
    node_bin = format(node, '09b')
    
    # Feature Index (2 bits)
    # Map hex feature to 2-bit index
    feature_raw = int(row['Feature']) if row['Feature'] != -1 else -1
    
    if feature_raw == -1:  # Leaf node
        feature_idx = 0b11  # Use 11 to mark leaf
    elif feature_raw == 0x00:
        feature_idx = 0b00
    elif feature_raw == 0x01:
        feature_idx = 0b01
    elif feature_raw == 0x10:
        feature_idx = 0b10
    else:
        print(f"WARNING: Unknown feature {feature_raw:02X}, using 00")
        feature_idx = 0b00
    
    feature_bin = format(feature_idx, '02b')
    
    # Threshold (64 bits) - quantized
    threshold = int(row['Threshold_Quant'])
    threshold_bin = format(threshold, '064b')
    
    # Left Child (9 bits)
    left = int(row['Left_Child'])
    if left >= 512:
        print(f"WARNING: Left child {left} exceeds 9-bit range!")
        left = 511
    left_bin = format(left, '09b')
    
    # Right Child (9 bits)
    right = int(row['Right_Child'])
    if right >= 512:
        print(f"WARNING: Right child {right} exceeds 9-bit range!")
        right = 511
    right_bin = format(right, '09b')
    
    # Prediction (2 bits)
    pred_raw = int(row['Prediction']) if row['Prediction'] != -1 else -1
    
    if pred_raw == -1:  # Internal node
        prediction = 0b00  # Default for internal nodes
    elif pred_raw == 0:
        prediction = 0b00  # Normal
    elif pred_raw == 1:
        prediction = 0b01  # Attack
    else:
        prediction = pred_raw & 0b11  # Clamp to 2 bits
    
    pred_bin = format(prediction, '02b')
    
    # Concatenate: [Node][Feature][Threshold][Left][Right][Pred]
    binary_str = node_bin + feature_bin + threshold_bin + left_bin + right_bin + pred_bin
    
    return binary_str

# ============= GENERATE OUTPUT =============
print("\n" + "="*80)
print("BINARY OUTPUT (95 bits per node)")
print("Format: Node(9)|Feature(2)|Threshold(64)|Left(9)|Right(9)|Pred(2)")
print("="*80)

binary_output = []
for idx, row in df.iterrows():
    binary_str = to_binary_95bit(row)
    binary_output.append(binary_str)
    
    # Print first 10 nodes with details
    if idx < 10:
        node = int(row['Node'])
        feat = row['Feature']
        thresh = row['Threshold_Quant']
        left = int(row['Left_Child'])
        right = int(row['Right_Child'])
        pred = row['Prediction']
        
        print(f"Node {node:3d}: {binary_str}")
        print(f"         Feature={feat if feat != -1 else 'LEAF':>4}, "
              f"Threshold={thresh}, Left={left}, Right={right}, Pred={pred}")

# Lưu file tree.mem
with open('tree.mem', 'w') as f:
    for line in binary_output:
        f.write(line + '\n')

print(f"\n✓ Generated {len(binary_output)} binary strings")
print(f"✓ Saved to 'tree.mem' (95-bit format)")

# ============= VERIFICATION =============
print("\n" + "="*80)
print("VERIFICATION - Decode node 0:")
print("="*80)

first = binary_output[0]
print(f"Binary string: {first}")
print(f"Total length:  {len(first)} bits")
print(f"\nBit breakdown:")
print(f"  [94:86] Node ID:    {first[0:9]} = {int(first[0:9], 2)}")
print(f"  [85:84] Feature:    {first[9:11]} = 0b{first[9:11]}")
print(f"  [83:20] Threshold:  {first[11:75]} = {int(first[11:75], 2)}")
print(f"  [19:11] Left Child: {first[75:84]} = {int(first[75:84], 2)}")
print(f"  [10:2]  Right Child:{first[84:93]} = {int(first[84:93], 2)}")
print(f"  [1:0]   Prediction: {first[93:95]} = 0b{first[93:95]}")

# ============= SAVE FEATURE MAPPING =============
print("\n" + "="*80)
print("FEATURE MAPPING (for reference):")
print("="*80)
print("  2'b00 (0) -> Feature 0x00 (Timestamp)")
print("  2'b01 (1) -> Feature 0x01 (Arbitration ID)")
print("  2'b10 (2) -> Feature 0x10 (Data Field)")
print("  2'b11 (3) -> LEAF NODE marker")

with open('feature_mapping.txt', 'w') as f:
    f.write("Feature Mapping for RTL:\n")
    f.write("========================\n")
    f.write("RTL mem_feature_idx [1:0] | Original Feature | Description\n")
    f.write("---------------------------|------------------|------------------\n")
    f.write("2'b00                      | 0x00             | Timestamp\n")
    f.write("2'b01                      | 0x01             | Arbitration ID\n")
    f.write("2'b10                      | 0x10             | Data Field\n")
    f.write("2'b11                      | 0xFF (leaf)      | Leaf Node\n")

print("\n✓ Feature mapping saved to 'feature_mapping.txt'")

# ============= GENERATE FEATURES.MEM FROM YOUR SAMPLE =============
print("\n" + "="*80)
print("Generating features.mem from your sample data:")
print("="*80)

# Your sample input
sample_input = {
    'timestamp': 1672531360.463805,
    'arbitration_id': '0C9',
    'data_field': "8416690D00000000",
}

# Parse values
timestamp_float = sample_input['timestamp']
arb_id_hex = int(sample_input['arbitration_id'], 16)
data_field_hex = int(sample_input['data_field'], 16)

# Extract first data byte (MSB of data_field)
first_data_byte = (data_field_hex >> 56) & 0xFF

print(f"\nInput Sample:")
print(f"  Timestamp:      {timestamp_float} seconds")
print(f"  Arbitration ID: 0x{arb_id_hex:03X} ({arb_id_hex})")
print(f"  Data Field:     0x{data_field_hex:016X}")
print(f"  First Byte:     0x{first_data_byte:02X} ({first_data_byte})")

# ============= CONVERT TO Q32.32 FORMAT =============
# Q32.32 Format: [63:32] integer part, [31:0] fractional part

# Feature 00: Timestamp (convert float to Q32.32)
timestamp_int = int(timestamp_float)  # Integer part
timestamp_frac = timestamp_float - timestamp_int  # Fractional part
timestamp_frac_scaled = int(timestamp_frac * (2**32))  # Scale to 32-bit

feature_00_q32 = (timestamp_int << 32) | timestamp_frac_scaled
feature_00_bin = format(feature_00_q32, '064b')

# Feature 01: Arbitration ID (integer, no fractional part)
feature_01_q32 = arb_id_hex << 32  # Integer part only
feature_01_bin = format(feature_01_q32, '064b')

# Feature 10: Data Field (use first byte as integer)
feature_10_q32 = first_data_byte << 32  # Integer part only
feature_10_bin = format(feature_10_q32, '064b')

# ============= SAVE TO FILE =============
with open('features.mem', 'w') as f:
    f.write(feature_00_bin + '\n')
    f.write(feature_01_bin + '\n')
    f.write(feature_10_bin + '\n')

print(f"\nQ32.32 Conversion:")
print(f"  Feature 00 (Timestamp):")
print(f"    Integer:    {timestamp_int} (0x{timestamp_int:08X})")
print(f"    Fractional: {timestamp_frac:.6f} -> {timestamp_frac_scaled} (0x{timestamp_frac_scaled:08X})")
print(f"    Q32.32:     0x{feature_00_q32:016X}")
print(f"    Binary:     {feature_00_bin}")
print(f"\n  Feature 01 (Arbitration ID):")
print(f"    Value:      {arb_id_hex} (0x{arb_id_hex:X})")
print(f"    Q32.32:     0x{feature_01_q32:016X}")
print(f"    Binary:     {feature_01_bin}")
print(f"\n  Feature 10 (First Data Byte):")
print(f"    Value:      {first_data_byte} (0x{first_data_byte:02X})")
print(f"    Q32.32:     0x{feature_10_q32:016X}")
print(f"    Binary:     {feature_10_bin}")

print("\n✓ Features saved to 'features.mem'")

# ============= VERIFICATION - DECODE BACK =============
print("\n" + "="*80)
print("VERIFICATION - Decode features.mem:")
print("="*80)

def decode_q32(binary_str):
    """Decode Q32.32 binary string back to float"""
    value = int(binary_str, 2)
    int_part = (value >> 32) & 0xFFFFFFFF
    frac_part = value & 0xFFFFFFFF
    float_value = int_part + (frac_part / (2**32))
    return float_value

with open('features.mem', 'r') as f:
    lines = f.readlines()
    
print(f"Feature 00: {lines[0].strip()}")
print(f"  Decoded: {decode_q32(lines[0].strip()):.6f} seconds")
print(f"  Original: {timestamp_float}")
print(f"  Match: {'✓' if abs(decode_q32(lines[0].strip()) - timestamp_float) < 0.0001 else '✗'}")

print(f"\nFeature 01: {lines[1].strip()}")
print(f"  Decoded: {int(lines[1].strip(), 2) >> 32} (0x{int(lines[1].strip(), 2) >> 32:X})")
print(f"  Original: {arb_id_hex} (0x{arb_id_hex:X})")
print(f"  Match: {'✓' if (int(lines[1].strip(), 2) >> 32) == arb_id_hex else '✗'}")

print(f"\nFeature 10: {lines[2].strip()}")
print(f"  Decoded: {int(lines[2].strip(), 2) >> 32} (0x{int(lines[2].strip(), 2) >> 32:X})")
print(f"  Original: {first_data_byte} (0x{first_data_byte:X})")
print(f"  Match: {'✓' if (int(lines[2].strip(), 2) >> 32) == first_data_byte else '✗'}")

print("\n" + "="*80)
print("COMPLETE! Files generated:")
print("  1. tree.mem            - 95-bit binary format for tree_memory.v")
print("  2. feature_mapping.txt - Feature index mapping reference")
print("  3. features.mem        - Sample input features for testbench")
print("="*80)
print("\nReady to use with your RTL!")