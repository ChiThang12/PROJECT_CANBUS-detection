// ============================================================================
// Simple Testbench for CAN Attack Detector
// Reads features from features.mem and outputs attack detection result
// ============================================================================

`timescale 1ns/1ps
`include "can_attack_detector_top.v"
module tb_can_attack_detector_top;

    // Clock and reset
    reg         clk;
    reg         rst_n;
    reg         start;
    
    // Outputs
    wire        done;
    wire        is_attack;
    wire [1:0]  attack_class;
    wire [8:0]  final_node;
    wire [4:0]  tree_depth;
    
    // ========================================================================
    // Clock generation - 100MHz (10ns period)
    // ========================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    can_attack_detector_top #(
        .TREE_DEPTH(512),
        .MAX_DEPTH(20),
        .TREE_FILE("tree.mem"),
        .FEATURE_FILE("features.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .is_attack(is_attack),
        .attack_class(attack_class),
        .final_node(final_node),
        .tree_depth(tree_depth)
    );
    
    // ========================================================================
    // Test Sequence
    // ========================================================================
    initial begin
        $display("\n");
        $display("========================================");
        $display("CAN Bus Attack Detection System");
        $display("========================================\n");
        
        // Initialize
        rst_n = 0;
        start = 0;
        
        // Reset pulse
        #20;
        rst_n = 1;
        $display("[TB] System reset at time %0t\n", $time);
        
        // Wait for features to load
        #50;
        
        // Start detection
        $display("[TB] Starting detection at time %0t\n", $time);
        start = 1;
        #10;
        start = 0;
        
        // Wait for completion
        wait(done);
        
        // Small delay to let display finish
        #20;
        
        // Final result summary
        $display("\n========================================");
        $display("FINAL RESULT SUMMARY");
        $display("========================================");
        $display("  Detection Status: %s", is_attack ? "ATTACK DETECTED" : "NORMAL TRAFFIC");
        $display("  Binary Output:    %b", is_attack);
        $display("  Classification:   %b", attack_class);
        $display("  Final Node ID:    %0d", final_node);
        $display("  Tree Depth:       %0d", tree_depth);
        $display("========================================\n");
        
        // Finish simulation
        #100;
        $display("[TB] Simulation completed at time %0t", $time);
        $finish;
    end
    
    // ========================================================================
    // Timeout watchdog (10us timeout)
    // ========================================================================
    initial begin
        #10000;
        $display("\n[ERROR] Simulation timeout!");
        $display("Detection did not complete within 10us");
        $finish;
    end
    
    // ========================================================================
    // VCD dump for waveform viewing
    // ========================================================================
    initial begin
        $dumpfile("can_attack_detector.vcd");
        $dumpvars(0, tb_can_attack_detector_top);
    end
    
    // ========================================================================
    // Monitor key signals
    // ========================================================================
    initial begin
        $monitor("Time=%0t | State=%0d | Node=%0d | Depth=%0d | Done=%b | Attack=%b", 
                 $time, dut.engine.state, dut.current_node, tree_depth, done, is_attack);
    end

endmodule