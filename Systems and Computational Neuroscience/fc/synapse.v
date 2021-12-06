`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/12/01 14:54:37
// Design Name: 
// Module Name: synapse
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module synapse(
input spike_in,
input [16-1:0]weight,
output [16-1:0]v_out
    );
assign v_out= spike_in ? weight : 16'd0;
endmodule

