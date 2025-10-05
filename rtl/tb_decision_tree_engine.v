`timescale 1ns/1ps
`include "decision_tree_engine.v"
module tb_decision_tree_engine;

    reg clk, rst_n;
    reg start;
    wire busy, done;

    // Input features (Q32.32)
    reg [63:0] feature_00;  // CAN ID
    reg [63:0] feature_01;  // DLC
    reg [63:0] feature_10;  // Data/Timestamp

    // Memory interface
    wire [8:0] mem_addr;
    reg  [8:0] mem_node_id;
    reg  [1:0] mem_feature_idx;
    reg  [63:0] mem_threshold;
    reg  [8:0] mem_left_child;
    reg  [8:0] mem_right_child;
    reg  [1:0] mem_prediction;
    reg        mem_is_leaf;
    reg        mem_data_valid;

    // Outputs
    wire [1:0] result;
    wire       is_attack;
    wire [8:0] final_node_id;

    // Debug
    wire [8:0] current_node;
    wire [4:0] tree_depth;
    wire [2:0] state;

    // Instantiate DUT
    decision_tree_engine dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .busy(busy), .done(done),
        .feature_00(feature_00),
        .feature_01(feature_01),
        .feature_10(feature_10),
        .mem_addr(mem_addr),
        .mem_node_id(mem_node_id),
        .mem_feature_idx(mem_feature_idx),
        .mem_threshold(mem_threshold),
        .mem_left_child(mem_left_child),
        .mem_right_child(mem_right_child),
        .mem_prediction(mem_prediction),
        .mem_is_leaf(mem_is_leaf),
        .mem_data_valid(mem_data_valid),
        .result(result),
        .is_attack(is_attack),
        .final_node_id(final_node_id),
        .current_node(current_node),
        .tree_depth(tree_depth),
        .state(state)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz
    end

    // Reset
    initial begin
        rst_n = 0;
        start = 0;
        feature_00 = 0;
        feature_01 = 0;
        feature_10 = 0;
        mem_node_id = 0;
        mem_feature_idx = 0;
        mem_threshold = 0;
        mem_left_child = 0;
        mem_right_child = 0;
        mem_prediction = 0;
        mem_is_leaf = 0;
        mem_data_valid = 0;
        #20;
        rst_n = 1;
    end

    // Fake memory model (simple decision tree)
    always @(posedge clk) begin
        mem_data_valid <= 0;

        case (mem_addr)
            // Root node (0)
            9'd0: begin
                mem_node_id     <= 9'd0;
                mem_feature_idx <= 2'b00;          // feature_00 (CAN ID)
                mem_threshold   <= 64'd100;        // threshold = 100
                mem_left_child  <= 9'd1;
                mem_right_child <= 9'd2;
                mem_prediction  <= 2'b00;
                mem_is_leaf     <= 0;
                mem_data_valid  <= 1;
            end

            // Node 1 (Leaf - Normal)
            9'd1: begin
                mem_node_id     <= 9'd1;
                mem_feature_idx <= 2'b00;
                mem_threshold   <= 64'd0;
                mem_left_child  <= 9'd0;
                mem_right_child <= 9'd0;
                mem_prediction  <= 2'b00;  // Normal
                mem_is_leaf     <= 1;
                mem_data_valid  <= 1;
            end

            // Node 2 (Leaf - Attack)
            9'd2: begin
                mem_node_id     <= 9'd2;
                mem_feature_idx <= 2'b00;
                mem_threshold   <= 64'd0;
                mem_left_child  <= 9'd0;
                mem_right_child <= 9'd0;
                mem_prediction  <= 2'b01;  // Attack
                mem_is_leaf     <= 1;
                mem_data_valid  <= 1;
            end
        endcase
    end

    // Stimulus
    initial begin
        @(posedge rst_n);
        #20;

        // Test case 1: CAN ID = 50 (<100) -> Normal
        feature_00 = 64'd50;
        feature_01 = 64'd8;
        feature_10 = 64'd123;
        start = 1; #10; start = 0;

        wait(done);
        $display("Test1: result=%b is_attack=%b final_node=%0d", result, is_attack, final_node_id);

        #50;

        // Test case 2: CAN ID = 200 (>100) -> Attack
        feature_00 = 64'd200;
        feature_01 = 64'd8;
        feature_10 = 64'd123;
        start = 1; #10; start = 0;

        wait(done);
        $display("Test2: result=%b is_attack=%b final_node=%0d", result, is_attack, final_node_id);

        #50;
        $finish;
    end

endmodule
