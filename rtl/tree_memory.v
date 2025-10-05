// ============================================================================
// OPTIMIZED Tree Memory - Reduced debug spam
// ============================================================================

module tree_memory #(
    parameter TREE_DEPTH = 512,
    parameter NODE_WIDTH = 95,
    parameter MEM_INIT_FILE = "tree.mem"
)(
    input               clk,
    input               rst_n,
    input               read_enable,
    input   [8:0]       node_addr,
    
    output  reg [8:0]   node_id,
    output  reg [1:0]   feature_idx,
    output  reg [63:0]  threshold,
    output  reg [8:0]   left_child,
    output  reg [8:0]   right_child,
    output  reg [1:0]   prediction,
    output  reg         is_leaf,
    output  reg         data_valid
);

    reg [NODE_WIDTH-1:0] tree_mem [0:TREE_DEPTH-1];
    reg [NODE_WIDTH-1:0] node_data;
    reg [8:0] prev_addr;  // ✅ Track previous address
    
    integer i;

    initial begin
        if (MEM_INIT_FILE != "") begin
            $readmemb(MEM_INIT_FILE, tree_mem);
            $display("[TREE_MEM] Initialized from %s", MEM_INIT_FILE);
        end else begin
            for (i = 0; i < TREE_DEPTH; i = i + 1) begin
                tree_mem[i] = {NODE_WIDTH{1'b0}};
            end
            $display("[TREE_MEM] Initialized with zeros");
        end
    end
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            node_data   <= {NODE_WIDTH{1'b0}};
            node_id     <= 9'b0;
            feature_idx <= 2'b0;
            threshold   <= 64'b0;
            left_child  <= 9'b0;
            right_child <= 9'b0;
            prediction  <= 2'b0;
            is_leaf     <= 1'b0;
            data_valid  <= 1'b0;
            prev_addr   <= 9'b0;
        end else begin
            if (read_enable) begin
                node_data   <= tree_mem[node_addr];
                
                // Parse node data
                node_id     <= tree_mem[node_addr][94:86];
                feature_idx <= tree_mem[node_addr][85:84];
                threshold   <= tree_mem[node_addr][83:20];
                left_child  <= tree_mem[node_addr][19:11];
                right_child <= tree_mem[node_addr][10:2];
                prediction  <= tree_mem[node_addr][1:0];
                
                is_leaf     <= (tree_mem[node_addr][19:11] == 9'b0) && 
                               (tree_mem[node_addr][10:2] == 9'b0);
                
                data_valid  <= 1'b1;
                
                // ✅ Only print when address changes
                if (node_addr != prev_addr) begin
                    $display("[TREE_MEM] Read node %0d: Feature[%b] Threshold=%h Left=%0d Right=%0d Leaf=%b Pred=%b",
                             tree_mem[node_addr][94:86],
                             tree_mem[node_addr][85:84],
                             tree_mem[node_addr][83:20],
                             tree_mem[node_addr][19:11],
                             tree_mem[node_addr][10:2],
                             (tree_mem[node_addr][19:11] == 9'b0) && (tree_mem[node_addr][10:2] == 9'b0),
                             tree_mem[node_addr][1:0]);
                    prev_addr <= node_addr;
                end
            end else begin
                data_valid <= 1'b0;
            end
        end
    end

endmodule