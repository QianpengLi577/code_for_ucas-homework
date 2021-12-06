`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/12/03 15:14:47
// Design Name: 
// Module Name: control
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

module generate_state(
clk,en_load,rst
    );
//generate 2 cycle load signal    3 cycle signal unload signal
//load-to-use time of BRAM is 2 cycle
input clk;
input rst;
output en_load;
reg load=0;
reg [3-1:0]count=0;
assign en_load = load;
always @(posedge clk or negedge rst )
begin

if (!rst) begin 
load <= 1'b0;
count<=2'b0;
end

else begin
count<=count+1;
if (count == 3'b001)
    load<=1'b1;
else if (count == 3'b011)begin
    load <= 1'b0;
end
else load <=load;

if (count == 3'b100) count<=3'b0;
end

end
endmodule

module generate_add(
en_load,addr_r,addr_w,flag,rst,data_1
);
// 20 simulation time step
input en_load;
output [6-1:0]addr_w;
input rst;
output data_1;
output [6-1:0]addr_r;
output flag;

reg [6-1:0]count_28=0;
reg [6-1:0]count_28_w=0;
reg [5-1:0]count_20=0;
reg flag_reg=0;

assign addr_r=count_28;
assign flag=flag_reg;
assign data_1 = !(count_20==5'd0);
assign addr_w=count_28_w;

always @(negedge en_load or negedge rst)
begin
if (!rst) begin
count_28<=0;
count_28_w<=0;
count_20<=0;
#60  flag_reg<=0;
end
else begin
if(count_28 === count_28_w)
count_28<=count_28+6'd1;
else begin
count_28<=count_28+6'd1;
count_28_w<=count_28_w+6'd1;
end
if(count_28>=6'd28-1) begin
count_28<=6'd0;
count_20<=count_20+5'd1;
end
if(count_28_w>=6'd28-1) count_28_w<=0;
if(count_20>=6'd20+1) begin
flag_reg<=1'b1;
end
end
end
endmodule