// ============================================================================
// CAN Bus Attack Detector - TOP Module (FIXED VERSION)
// Integrates all components for complete attack detection system
// ============================================================================

`timescale 1ns/1ps
`include "feature_extractor.v"
`include "tree_memory.v"
`include "decision_tree_engine.v"


module can_attack_detector_top #(
    parameter TREE_DEPTH = 512,
    parameter MAX_DEPTH = 50,
    parameter TREE_FILE = "tree.mem",
    parameter FEATURE_FILE = "features.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,           // Pulse để bắt đầu detection
    
    output wire        done,            // Detection hoàn thành
    output wire        is_attack,       // 1=Attack, 0=Normal
    output wire [1:0]  attack_class,    // Classification result
    output wire [8:0]  final_node,      // Final leaf node
    output wire [4:0]  tree_depth       // Tree depth reached
);

    // ========================================================================
    // Load features từ file
    // ========================================================================
    reg [63:0] features [0:2];  // 3 features
    
    initial begin
        if (FEATURE_FILE != "") begin
            $readmemb(FEATURE_FILE, features);
            $display("\n[WRAPPER] Features loaded from %s:", FEATURE_FILE);
            $display("  Feature 00 (Timestamp):      %h", features[0]);
            $display("  Feature 01 (Arbitration ID): %h", features[1]);
            $display("  Feature 10 (Data Field):     %h\n", features[2]);
        end
    end
    
    wire [63:0] feature_00 = features[0];
    wire [63:0] feature_01 = features[1];
    wire [63:0] feature_10 = features[2];
    
    // ========================================================================
    // Tree Memory Interface
    // ========================================================================
    wire [8:0]  mem_addr;
    wire [8:0]  mem_node_id;
    wire [1:0]  mem_feature_idx;
    wire [63:0] mem_threshold;
    wire [8:0]  mem_left_child;
    wire [8:0]  mem_right_child;
    wire [1:0]  mem_prediction;
    wire        mem_is_leaf;
    wire        mem_data_valid;
    
    // ========================================================================
    // Engine outputs
    // ========================================================================
    wire        engine_busy;
    wire [8:0]  current_node;
    wire [2:0]  engine_state;
    
    // ========================================================================
    // Tree Memory Instance
    // ========================================================================
    tree_memory #(
        .TREE_DEPTH(TREE_DEPTH),
        .NODE_WIDTH(95),
        .MEM_INIT_FILE(TREE_FILE)
    ) tree_mem (
        .clk(clk),
        .rst_n(rst_n),
        .read_enable(1'b1),
        .node_addr(mem_addr),
        .node_id(mem_node_id),
        .feature_idx(mem_feature_idx),
        .threshold(mem_threshold),
        .left_child(mem_left_child),
        .right_child(mem_right_child),
        .prediction(mem_prediction),
        .is_leaf(mem_is_leaf),
        .data_valid(mem_data_valid)
    );
    
    // ========================================================================
    // Decision Tree Engine Instance
    // ========================================================================
    decision_tree_engine #(
        .MAX_DEPTH(MAX_DEPTH)
    ) engine (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(engine_busy),
        .done(done),
        
        // Features
        .feature_00(feature_00),
        .feature_01(feature_01),
        .feature_10(feature_10),
        
        // Memory interface
        .mem_addr(mem_addr),
        .mem_node_id(mem_node_id),
        .mem_feature_idx(mem_feature_idx),
        .mem_threshold(mem_threshold),
        .mem_left_child(mem_left_child),
        .mem_right_child(mem_right_child),
        .mem_prediction(mem_prediction),
        .mem_is_leaf(mem_is_leaf),
        .mem_data_valid(mem_data_valid),
        
        // Outputs
        .result(attack_class),
        .is_attack(is_attack),
        .final_node_id(final_node),
        .current_node(current_node),
        .tree_depth(tree_depth),
        .state(engine_state)
    );
    
    // ========================================================================
    // Result Display
    // ========================================================================
    always @(posedge clk) begin
        if (done) begin
            $display("\n========================================");
            $display("DETECTION RESULT:");
            $display("  Status:      %s", is_attack ? "⚠️  ATTACK DETECTED" : "✅ NORMAL");
            $display("  Class:       %b", attack_class);
            $display("  Final Node:  %0d", final_node);
            $display("  Tree Depth:  %0d", tree_depth);
            $display("========================================\n");
        end
    end

endmodule