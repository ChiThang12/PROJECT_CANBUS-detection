// ============================================================================
// Tree Reader Module
// ============================================================================
module tree_decoder #(
    parameter DEPTH = 64   // số node tối đa (tùy file .mem của bạn)
)(
    input  wire clk,
    input  wire rst_n
);

    // Mỗi dòng 95 bit
    reg [94:0] mem [0:DEPTH-1];

    // Đọc file memory
    initial begin
        $readmemb("tree.mem", mem);
    end

    // Một số biến để debug
    integer i;
    reg [8:0] node;
    reg [1:0] feature;
    reg [63:0] threshold;
    reg [8:0] left_child;
    reg [8:0] right_child;
    reg [1:0] prediction;

    // Tách field từ frame
    task decode_frame(input [94:0] frame);
    begin
        node       = frame[94:86];   // 9 bit
        feature    = frame[85:84];   // 2 bit
        threshold  = frame[83:20];   // 64 bit
        left_child = frame[19:11];   // 9 bit
        right_child= frame[10:2];    // 9 bit
        prediction = frame[1:0];     // 2 bit
    end
    endtask

    // In ra dữ liệu khi reset done
    always @(posedge clk) begin
        if (!rst_n) begin
            // nothing
        end else begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                decode_frame(mem[i]);
                $display("Node=%0d | Feature=%0d | Threshold=%0d | Left=%0d | Right=%0d | Pred=%0b",
                         node, feature, threshold, left_child, right_child, prediction);
            end
            $finish;
        end
    end

endmodule

`timescale 1ns/1ps

module tb_tree_reader;
    reg clk;
    reg rst_n;

    // Instance
    tree_decoder #(.DEPTH(32)) uut (
        .clk(clk),
        .rst_n(rst_n)
    );

    // Clock 10ns = 100MHz
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Reset logic
    initial begin
        rst_n = 0;
        #20;
        rst_n = 1;
    end

endmodule

