`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/12/01 14:57:17
// Design Name: 
// Module Name: neuron
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

module neuron_act_fc(clk,v_in,s_old,s_out,v_old,v_out);
// clk is clock   1bit   v_in is result of fc  5.11 16bit
// s_old is old spike from BRAM  s_out is new spike to store in BRAM   1bit
// v_old is old v from BRAM   v_out is new v to stroe in BRAM       5.11 16bit
input clk;
input [16*10-1:0]v_in;
input [16*10-1:0]v_old;
output [16*10-1:0]v_out;
input [10-1:0]s_old;
output [10-1:0]s_out;

// decay is 0.25    that means mult deacy just >>> 2  but operation >>> doesnt work well  thus rewrite the >>>
// threshold  is 0.5
generate 
    genvar i;
    for(i=0;i<10;i=i+1)begin
        act aa(clk,v_in[16*(i+1)-1:16*i],s_old[i],s_out[i],v_old[16*(i+1)-1:16*i],v_out[16*(i+1)-1:16*i]);
    end
endgenerate

endmodule

module act(clk,v_in,s_old,s_out,v_old,v_out);
// clk is clock   1bit   v_in is result of fc  5.11 16bit
// s_old is old spike from BRAM  s_out is new spike to store in BRAM   1bit
// v_old is old v from BRAM   v_out is new v to stroe in BRAM       5.11 16bit
input clk;
input signed[16-1:0] v_in;
input s_old;
input signed[16-1:0] v_old;
output s_out;
output signed[16-1:0] v_out;
// decay is 0.25    that means mult deacy just >>> 2  but operation >>> doesnt work well  thus rewrite the >>>
// threshold  is 0.5
reg  s_temp;
wire sign ;
assign sign = v_old[15];
//assign v_out=v_in + {16{~s_old}}&(v_old>>>2); // it does not work well
assign v_out=v_in + $signed({16{~s_old}}&({sign,sign,v_old[15:2]}));
assign s_out=s_temp;
always@(*)
if (v_out>=16'sb0000010000000000)   s_temp<=1'b1;  //threshold is 0.5
else s_temp<=1'b0;
endmodule

module add_v_fc(clk,rst,v_in,addr_r,addr_w,v_out,b,en);
// this part is sum the neuron's v   using addr_r and addr_w to ensure bias is sum correctly
input clk;
input rst;
input signed[16*10-1:0]v_in;
input [6-1:0]addr_r;
input [6-1:0]addr_w;
input signed[16*10-1:0]b;
output signed[16*10-1:0]v_out;
output en;
reg [16*10-1:0]v_temp;
reg en_reg;
assign en=en_reg;
assign v_out=v_temp;
always@(posedge clk or negedge rst)
begin
if(!rst) begin
 v_temp<=0;
en_reg<=1'b0;
 end
else begin
if(addr_r==6'b000000 & addr_w==6'b011011) 
begin
v_temp[15 : 0]<=v_in[15 : 0]+v_temp[15 : 0]+b[15 : 0];
v_temp[31 : 16]<=v_in[31 : 16]+v_temp[31 : 16]+b[31 : 16];
v_temp[47 : 32]<=v_in[47 : 32]+v_temp[47 : 32]+b[47 : 32];
v_temp[63 : 48]<=v_in[63 : 48]+v_temp[63 : 48]+b[63 : 48];
v_temp[79 : 64]<=v_in[79 : 64]+v_temp[79 : 64]+b[79 : 64];
v_temp[95 : 80]<=v_in[95 : 80]+v_temp[95 : 80]+b[95 : 80];
v_temp[111 : 96]<=v_in[111 : 96]+v_temp[111 : 96]+b[111 : 96];
v_temp[127 : 112]<=v_in[127 : 112]+v_temp[127 : 112]+b[127 : 112];
v_temp[143 : 128]<=v_in[143 : 128]+v_temp[143 : 128]+b[143 : 128];
v_temp[159 : 144]<=v_in[159 : 144]+v_temp[159 : 144]+b[159 : 144];
en_reg<=1'b1;
end
else if (addr_r==6'b000001 & addr_w==6'b000000)  // 
begin
en_reg<=1'b0;
v_temp[15 : 0]<=v_in[15 : 0];
v_temp[31 : 16]<=v_in[31 : 16];
v_temp[47 : 32]<=v_in[47 : 32];
v_temp[63 : 48]<=v_in[63 : 48];
v_temp[79 : 64]<=v_in[79 : 64];
v_temp[95 : 80]<=v_in[95 : 80];
v_temp[111 : 96]<=v_in[111 : 96];
v_temp[127 : 112]<=v_in[127 : 112];
v_temp[143 : 128]<=v_in[143 : 128];
v_temp[159 : 144]<=v_in[159 : 144];
end
else begin
v_temp[15 : 0]<=v_in[15 : 0]+v_temp[15 : 0];
v_temp[31 : 16]<=v_in[31 : 16]+v_temp[31 : 16];
v_temp[47 : 32]<=v_in[47 : 32]+v_temp[47 : 32];
v_temp[63 : 48]<=v_in[63 : 48]+v_temp[63 : 48];
v_temp[79 : 64]<=v_in[79 : 64]+v_temp[79 : 64];
v_temp[95 : 80]<=v_in[95 : 80]+v_temp[95 : 80];
v_temp[111 : 96]<=v_in[111 : 96]+v_temp[111 : 96];
v_temp[127 : 112]<=v_in[127 : 112]+v_temp[127 : 112];
v_temp[143 : 128]<=v_in[143 : 128]+v_temp[143 : 128];
v_temp[159 : 144]<=v_in[159 : 144]+v_temp[159 : 144];
end

end


end
endmodule