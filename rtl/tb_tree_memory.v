`timescale 1ns/1ps
`include "tree_memory.v"
module tb_tree_memory;

    // Clock và reset
    reg clk;
    reg rst_n;
    reg read_enable;
    reg [8:0] node_addr;

    // Outputs từ DUT
    wire [8:0]  node_id;
    wire [1:0]  feature_idx;
    wire [63:0] threshold;
    wire [8:0]  left_child;
    wire [8:0]  right_child;
    wire [1:0]  prediction;
    wire        is_leaf;
    wire        data_valid;

    // Instantiate DUT
    tree_memory #(
        .TREE_DEPTH(512),
        .NODE_WIDTH(95),
        .MEM_INIT_FILE("tree.mem")   // file tree.mem có trong cùng folder
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .read_enable(read_enable),
        .node_addr(node_addr),
        .node_id(node_id),
        .feature_idx(feature_idx),
        .threshold(threshold),
        .left_child(left_child),
        .right_child(right_child),
        .prediction(prediction),
        .is_leaf(is_leaf),
        .data_valid(data_valid)
    );

    // Clock 10ns (100MHz)
    always #5 clk = ~clk;

    // Test logic
    initial begin
        $display("==== Start Tree Memory Testbench ====");
        clk = 0;
        rst_n = 0;
        read_enable = 0;
        node_addr = 0;

        // Reset
        #20;
        rst_n = 1;

        // Đọc lần lượt vài node trong tree
        #10;
        read_enable = 1;
        node_addr = 0;   // đọc node gốc
        #10;
        node_addr = 1;   // đọc node 1
        #10;
        node_addr = 2;   // đọc node 2
        #10;
        node_addr = 3;   // đọc node 3
        #10;
        node_addr = 4;   // đọc node 4
        #10;

        // Tắt read_enable
        read_enable = 0;

        // Dừng mô phỏng
        #50;
        $display("==== Finish Testbench ====");
        $finish;
    end

    // Monitor để in dữ liệu node (ngoài $display trong DUT)
    always @(posedge clk) begin
        if (data_valid) begin
            $display("[TB] t=%0t | Node=%0d | Feature=%b | Threshold=%h | Left=%0d | Right=%0d | Pred=%0d | Leaf=%b",
                $time, node_id, feature_idx, threshold, left_child, right_child, prediction, is_leaf);
        end
    end

endmodule
