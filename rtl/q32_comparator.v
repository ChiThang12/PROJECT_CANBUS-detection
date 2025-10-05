// ============================================================================
// Q32.32 Fixed-Point Comparator for Decision Tree
// Simple comparator: feature <= threshold (for tree branching)
// 
// Q32.32 Format:
//   - Bits [63:32]: Signed integer part (32-bit)
//   - Bits [31:0]:  Unsigned fractional part (32-bit)
//
// Usage in Decision Tree:
//   if (feature <= threshold) go to left_child
//   else                      go to right_child
// ============================================================================


module q32_comparator (
    input  wire [63:0] feature,      // Feature value (Q32.32)
    input  wire [63:0] threshold,    // Threshold value (Q32.32)
    output wire        go_left       // 1 = feature <= threshold (go left)
                                     // 0 = feature > threshold (go right)
);

    // Extract integer and fractional parts
    wire signed [31:0] feature_int;
    wire        [31:0] feature_frac;
    wire signed [31:0] threshold_int;
    wire        [31:0] threshold_frac;
    
    assign feature_int    = feature[63:32];
    assign feature_frac   = feature[31:0];
    assign threshold_int  = threshold[63:32];
    assign threshold_frac = threshold[31:0];
    
    // Compare logic
    wire int_less;
    wire int_equal;
    wire frac_less_or_equal;
    
    // Compare integer parts (signed comparison)
    assign int_less  = (feature_int < threshold_int);
    assign int_equal = (feature_int == threshold_int);
    
    // Compare fractional parts (unsigned comparison)
    assign frac_less_or_equal = (feature_frac <= threshold_frac);
    
    // Final result: feature <= threshold
    // True if:
    //   1. Integer part of feature < threshold, OR
    //   2. Integer parts equal AND fractional part <= threshold
    assign go_left = int_less || (int_equal && frac_less_or_equal);

endmodule