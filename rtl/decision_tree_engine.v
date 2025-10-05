// ============================================================================
// OPTIMIZED Decision Tree Engine - Fixed memory read spam issue
// ============================================================================

`timescale 1ns/1ps
`include "q32_comparator.v"
module decision_tree_engine #(
    parameter MAX_DEPTH = 20
)(
    input  wire        clk,
    input  wire        rst_n,
    
    input  wire        start,
    output reg         busy,
    output reg         done,
    
    input  wire [63:0] feature_00,
    input  wire [63:0] feature_01,
    input  wire [63:0] feature_10,
    
    output reg  [8:0]  mem_addr,
    input  wire [8:0]  mem_node_id,
    input  wire [1:0]  mem_feature_idx,
    input  wire [63:0] mem_threshold,
    input  wire [8:0]  mem_left_child,
    input  wire [8:0]  mem_right_child,
    input  wire [1:0]  mem_prediction,
    input  wire        mem_is_leaf,
    input  wire        mem_data_valid,
    
    output reg  [1:0]  result,
    output reg         is_attack,
    output reg  [8:0]  final_node_id,
    
    output reg  [8:0]  current_node,
    output reg  [4:0]  tree_depth,
    output reg  [2:0]  state
);

    localparam [2:0] IDLE           = 3'd0;
    localparam [2:0] LOAD_NODE      = 3'd1;
    localparam [2:0] WAIT_MEM       = 3'd2;
    localparam [2:0] COMPARE        = 3'd3;
    localparam [2:0] DECIDE         = 3'd4;
    localparam [2:0] OUTPUT_RESULT  = 3'd5;
    localparam [2:0] ERROR          = 3'd6;
    
    reg [63:0] selected_feature;
    reg        comparison_result;
    reg [1:0]  wait_counter;  // ✅ Counter để chờ memory
    
    wire go_left;
    
    q32_comparator comparator (
        .feature(selected_feature),
        .threshold(mem_threshold),
        .go_left(go_left)
    );
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state             <= IDLE;
            busy              <= 1'b0;
            done              <= 1'b0;
            mem_addr          <= 9'd0;
            result            <= 2'b00;
            is_attack         <= 1'b0;
            final_node_id     <= 9'd0;
            current_node      <= 9'd0;
            tree_depth        <= 5'd0;
            selected_feature  <= 64'b0;
            comparison_result <= 1'b0;
            wait_counter      <= 2'd0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy         <= 1'b1;
                        mem_addr     <= 9'd0;
                        current_node <= 9'd0;
                        tree_depth   <= 5'd0;
                        wait_counter <= 2'd0;
                        state        <= LOAD_NODE;
                        $display("\n[ENGINE] Starting inference from root node");
                    end
                end
                
                LOAD_NODE: begin
                    mem_addr     <= current_node;
                    wait_counter <= 2'd0;
                    state        <= WAIT_MEM;
                end
                
                WAIT_MEM: begin
                    // ✅ Chờ 2 cycles cho memory ổn định
                    if (wait_counter < 2'd2) begin
                        wait_counter <= wait_counter + 1'd1;
                    end else if (mem_data_valid) begin
                        if (mem_is_leaf) begin
                            result        <= mem_prediction;
                            is_attack     <= (mem_prediction == 2'b01);
                            final_node_id <= mem_node_id;
                            state         <= OUTPUT_RESULT;
                            $display("[ENGINE] Reached LEAF node=%0d, prediction=%b", 
                                     mem_node_id, mem_prediction);
                        end else begin
                            state <= COMPARE;
                        end
                    end
                end
                
                COMPARE: begin
                    // Chọn feature
                    case (mem_feature_idx)
                        2'b00: selected_feature <= feature_00;
                        2'b01: selected_feature <= feature_01;
                        2'b10: selected_feature <= feature_10;
                        default: selected_feature <= 64'b0;
                    endcase
                    
                    comparison_result <= go_left;
                    
                    $display("[ENGINE] Node=%0d Depth=%0d Feature[%b]=%h <= Threshold=%h ? %b",
                             mem_node_id, tree_depth, mem_feature_idx, 
                             selected_feature, mem_threshold, go_left);
                    
                    state <= DECIDE;
                end
                
                DECIDE: begin
                    // ✅ Gán trực tiếp child node
                    if (comparison_result) begin
                        current_node <= mem_left_child;
                        $display("[ENGINE] Going LEFT to node %0d", mem_left_child);
                    end else begin
                        current_node <= mem_right_child;
                        $display("[ENGINE] Going RIGHT to node %0d", mem_right_child);
                    end
                    
                    tree_depth <= tree_depth + 1;
                    
                    if (tree_depth >= MAX_DEPTH) begin
                        $display("ERROR: Max depth %0d reached!", MAX_DEPTH);
                        state <= ERROR;
                    end else begin
                        state <= LOAD_NODE;
                    end
                end
                
                OUTPUT_RESULT: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    $display("\n[ENGINE] ✅ INFERENCE COMPLETE:");
                    $display("         Result: %s (class=%b)", 
                             is_attack ? "ATTACK" : "NORMAL", result);
                    $display("         Final node: %0d", final_node_id);
                    $display("         Tree depth: %0d\n", tree_depth);
                    state <= IDLE;
                end
                
                ERROR: begin
                    result        <= 2'b00;
                    is_attack     <= 1'b0;
                    final_node_id <= 9'd0;
                    done          <= 1'b1;
                    busy          <= 1'b0;
                    state         <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule