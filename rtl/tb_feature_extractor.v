`timescale 1ns/1ps
`include "feature_extractor.v"
module feature_extractor_tb;

    reg clk;
    reg rst_n;

    // CAN frame inputs
    reg  [28:0] can_id;
    reg  [3:0]  can_dlc;
    reg  [63:0] can_data;
    reg         frame_valid;
    reg         frame_extended;

    // Outputs
    wire [63:0] feature_00;
    wire [63:0] feature_01;
    wire [63:0] feature_10;
    wire        features_ready;
    wire [2:0]  state;
    wire        busy;

    // Instantiate DUT
    feature_extractor dut (
        .clk(clk),
        .rst_n(rst_n),
        .can_id(can_id),
        .can_dlc(can_dlc),
        .can_data(can_data),
        .frame_valid(frame_valid),
        .frame_extended(frame_extended),
        .feature_00(feature_00),
        .feature_01(feature_01),
        .feature_10(feature_10),
        .features_ready(features_ready),
        .state(state),
        .busy(busy)
    );

    // Clock generation: 100MHz -> 10ns period
    always #5 clk = ~clk;

    // Stimulus
    initial begin
        // Dump VCD file
        $dumpfile("feature_extractor_tb.vcd");
        $dumpvars(0, feature_extractor_tb);

        // Init
        clk = 0;
        rst_n = 0;
        frame_valid = 0;
        can_id = 0;
        can_dlc = 0;
        can_data = 0;
        frame_extended = 0;

        // Reset
        #20;
        rst_n = 1;

        // Wait a bit
        #20;

        // Apply first CAN frame
        @(posedge clk);
        frame_valid = 1;
        can_id = 29'h000001AB;                // Example ID
        can_dlc = 4'd8;                       // DLC = 8
        can_data = 64'h8416690D00000000;      // Example data
        frame_extended = 1;                   // Extended frame

        @(posedge clk);
        frame_valid = 0;   // Only pulse 1 cycle

        // Wait until features_ready
        wait(features_ready == 1);

        // Hold for some time to see DONE state
        #50;

        // Apply another CAN frame (standard ID)
        @(posedge clk);
        frame_valid = 1;
        can_id = 29'h0000007A;                // Standard ID
        can_dlc = 4'd4;                       // DLC = 4
        can_data = 64'h1122334455667788;      // Data
        frame_extended = 0;                   // Standard frame

        @(posedge clk);
        frame_valid = 0;

        wait(features_ready == 1);
        #50;

        // Finish simulation
        $finish;
    end

endmodule
