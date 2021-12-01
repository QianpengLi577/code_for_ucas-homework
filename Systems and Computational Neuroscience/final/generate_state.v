`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/11/29 14:51:42
// Design Name: 
// Module Name: generate_state
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
count<=3'b0;
end

else begin
count<=count+1;
if (count == 3'b001)
    load<=1'b1;
else if (count == 3'b011)
    load <= 1'b0;
else load <=load;
end

end
endmodule

module generate_add(
en_load,addr_w,addr_r,flag,rst,data_1
);
input en_load;
input rst;
output data_1;
output [4-1:0]addr_w;
output [4-1:0]addr_r;
output flag;
reg [4-1:0]w=0;
reg [4-1:0]r=0;
reg [10-1:0]count=0;
reg flag_reg=0;
reg data_1_reg=0;
assign addr_w=w;
assign addr_r=r;
assign flag=flag_reg;
assign data_1 = data_1_reg;
always @(negedge en_load or negedge rst)
begin
if (!rst) begin
count<=0;
r<=0;
w<=0;
data_1_reg<=0;
#60  flag_reg<=0;
end
else begin
count<=count+1;
if (w===r)
r<=r+1;
else begin
r<=r+1;
w<=w+1;
end
if (count==10'd320)
flag_reg=1'b1;
if (count <10'd16)
data_1_reg=1'b0;
else 
data_1_reg=1'b1;
end

end
endmodule

