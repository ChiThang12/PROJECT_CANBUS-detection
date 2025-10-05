// ============================================================================
// Feature Extractor Module for CAN Bus Attack Detection
// Extracts features from CAN frame and converts to Q32.32 fixed-point format
// 
// Feature Mapping:
//   - Feature 00: Timestamp (time when frame arrived)
//   - Feature 01: Arbitration ID (CAN ID)
//   - Feature 10: Data Field (first data byte or combined data)
// ============================================================================

`timescale 1ns/1ps

module feature_extractor #(
    parameter TIMESTAMP_WIDTH = 64  // 64-bit timestamp
)(
    input  wire         clk,
    input  wire         rst_n,
    
    // CAN Frame Input Interface
    input  wire [TIMESTAMP_WIDTH-1:0] timestamp,     // Frame timestamp (e.g., 1672531360.463805)
    input  wire [28:0]  arbitration_id,              // CAN Arbitration ID (e.g., 0x0C9)
    input  wire [63:0]  data_field,                  // 8 bytes data (e.g., 0x8416690D00000000)
    input  wire         frame_valid,                 // Pulse when new frame arrives
    
    // Extracted Features Output (Q32.32 format)
    output reg  [63:0]  feature_00,      // Timestamp as Q32.32
    output reg  [63:0]  feature_01,      // Arbitration ID as Q32.32
    output reg  [63:0]  feature_10,      // Data field as Q32.32
    output reg          features_ready,  // Features are valid and ready
    
    // Status outputs
    output reg  [2:0]   state,           // Current FSM state (for debug)
    output reg          busy             // Module is processing
);

    // State machine states
    localparam [2:0] IDLE         = 3'd0;
    localparam [2:0] CAPTURE      = 3'd1;
    localparam [2:0] CONVERT      = 3'd2;
    localparam [2:0] DONE         = 3'd3;
    
    // Internal registers to capture frame
    reg [TIMESTAMP_WIDTH-1:0] captured_timestamp;
    reg [28:0]  captured_arbitration_id;
    reg [63:0]  captured_data_field;
    
    // Previous timestamp for time delta calculation
    reg [TIMESTAMP_WIDTH-1:0] prev_timestamp;
    reg         first_frame;
    
    // ========================================================================
    // Main FSM for feature extraction
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                   <= IDLE;
            captured_timestamp      <= {TIMESTAMP_WIDTH{1'b0}};
            captured_arbitration_id <= 29'b0;
            captured_data_field     <= 64'b0;
            prev_timestamp          <= {TIMESTAMP_WIDTH{1'b0}};
            first_frame             <= 1'b1;
            feature_00              <= 64'b0;
            feature_01              <= 64'b0;
            feature_10              <= 64'b0;
            features_ready          <= 1'b0;
            busy                    <= 1'b0;
        end else begin
            case (state)
                // ============================================================
                IDLE: begin
                    features_ready <= 1'b0;
                    busy <= 1'b0;
                    
                    if (frame_valid) begin
                        // Capture incoming frame
                        captured_timestamp      <= timestamp;
                        captured_arbitration_id <= arbitration_id;
                        captured_data_field     <= data_field;
                        busy                    <= 1'b1;
                        state                   <= CAPTURE;
                    end
                end
                
                // ============================================================
                CAPTURE: begin
                    // Frame captured, start conversion
                    state <= CONVERT;
                end
                
                // ============================================================
                CONVERT: begin
                    // Feature 00: Timestamp
                    // Option 1: Use absolute timestamp
                    // Option 2: Use time delta from previous frame (for attack detection)
                    if (first_frame) begin
                        // First frame - use absolute timestamp (lower 64 bits)
                        feature_00 <= captured_timestamp[63:0];
                        first_frame <= 1'b0;
                    end else begin
                        // Calculate time delta: current - previous
                        // This is critical for temporal attack detection
                        feature_00 <= captured_timestamp[63:0] - prev_timestamp[63:0];
                    end
                    
                    // Feature 01: Arbitration ID
                    // Convert CAN ID to Q32.32 format
                    // Most CAN IDs are 11-bit (standard) or 29-bit (extended)
                    feature_01 <= {35'b0, captured_arbitration_id};  // Integer part only
                    
                    // Feature 10: Data Field
                    // Option 1: Extract first byte
                    // Option 2: Extract specific bytes based on CAN protocol
                    // Option 3: Use all 64 bits directly
                    
                    // Using first data byte (MSB)
                    feature_10 <= {56'b0, captured_data_field[63:56]};
                    
                    // Alternative: Use all data as 64-bit value
                    // feature_10 <= captured_data_field;
                    
                    // Update previous timestamp
                    prev_timestamp <= captured_timestamp;
                    
                    state <= DONE;
                end
                
                // ============================================================
                DONE: begin
                    features_ready <= 1'b1;
                    busy <= 1'b0;
                    // Stay in DONE for one cycle, then return to IDLE
                    state <= IDLE;
                end
                
                // ============================================================
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end
    
    // ========================================================================
    // Debug display
    // ========================================================================
    always @(posedge clk) begin
        if (frame_valid) begin
            $display("\n[FEATURE_EXTRACTOR] Time=%0t | New CAN Frame:", $time);
            $display("  Timestamp:      %0d (0x%h)", timestamp, timestamp);
            $display("  Arbitration ID: 0x%h", arbitration_id);
            $display("  Data Field:     0x%h", data_field);
        end
        
        if (features_ready) begin
            $display("\n[FEATURE_EXTRACTOR] Time=%0t | Features Extracted:", $time);
            $display("  Feature 00 (Timestamp):  0x%h", feature_00);
            $display("  Feature 01 (Arb ID):     0x%h", feature_01);
            $display("  Feature 10 (Data):       0x%h", feature_10);
        end
    end

endmodule