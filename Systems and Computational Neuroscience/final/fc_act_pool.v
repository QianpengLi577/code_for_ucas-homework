`timescale 1ns / 1ns
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/11/24 08:49:24
// Design Name: 
// Module Name: fc_act_pool
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
//   12*10 full connection layer
//   x(spike passing pooling layer) 1.2 3bit
//   w(full connection layer weight) 3.13 16bit
//   b(full connection layer weight biasc) 3.13 16bit
//   out 3.13 16bit
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module fc_act_pool(
clk,x,w,out
    );
input clk;
input [3*7*7-1:0]x;
input signed[7*7*10*16-1:0]w;
//input signed[10*16-1:0]b;
output signed[10*16-1:0]out;

wire signed[7*7*10*16-1:0]out_temp;

//get spike mul weight
generate 
        genvar i,j;
        for(i = 0; i < 10; i = i + 1) begin
            for(j=0; j < 49; j = j + 1)begin
                  mul_conv_fc mul_name (
                      .A(w[16*49*i+16*(j+1)-1:16*49*i+16*j]),      // input wire [15 : 0] A
                      .B(x[3*(j+1)-1:3*j]),      // input wire [2 : 0] B
                      .P(out_temp[16*49*i+16*(j+1)-1:16*49*i+16*j])      // output wire [15 : 0] P
                    );      
            end
        end
endgenerate 

//get result of add
generate 
        genvar k;
        for(k = 0;k < 10; k = k + 1) begin
            assign out[16*(k+1)-1:16*k]=out_temp[16*49*k+16*(1)-1:16*49*k+16*0]
                                                        +out_temp[16*49*k+16*(2)-1:16*49*k+16*1]
                                                        +out_temp[16*49*k+16*(3)-1:16*49*k+16*2]
                                                        +out_temp[16*49*k+16*(4)-1:16*49*k+16*3]
                                                        +out_temp[16*49*k+16*(5)-1:16*49*k+16*4]
                                                        +out_temp[16*49*k+16*(6)-1:16*49*k+16*5]
                                                        +out_temp[16*49*k+16*(7)-1:16*49*k+16*6]
                                                        +out_temp[16*49*k+16*(8)-1:16*49*k+16*7]
                                                        +out_temp[16*49*k+16*(9)-1:16*49*k+16*8]
                                                        +out_temp[16*49*k+16*(10)-1:16*49*k+16*9]
                                                        +out_temp[16*49*k+16*(11)-1:16*49*k+16*10]
                                                        +out_temp[16*49*k+16*(12)-1:16*49*k+16*11]
                                                        +out_temp[16*49*k+16*(13)-1:16*49*k+16*12]
                                                        +out_temp[16*49*k+16*(14)-1:16*49*k+16*13]
                                                        +out_temp[16*49*k+16*(15)-1:16*49*k+16*14]
                                                        +out_temp[16*49*k+16*(16)-1:16*49*k+16*15]
                                                        +out_temp[16*49*k+16*(17)-1:16*49*k+16*16]
                                                        +out_temp[16*49*k+16*(18)-1:16*49*k+16*17]
                                                        +out_temp[16*49*k+16*(19)-1:16*49*k+16*18]
                                                        +out_temp[16*49*k+16*(20)-1:16*49*k+16*19]
                                                        +out_temp[16*49*k+16*(21)-1:16*49*k+16*20]
                                                        +out_temp[16*49*k+16*(22)-1:16*49*k+16*21]
                                                        +out_temp[16*49*k+16*(23)-1:16*49*k+16*22]
                                                        +out_temp[16*49*k+16*(24)-1:16*49*k+16*23]
                                                        +out_temp[16*49*k+16*(25)-1:16*49*k+16*24]
                                                        +out_temp[16*49*k+16*(26)-1:16*49*k+16*25]
                                                        +out_temp[16*49*k+16*(27)-1:16*49*k+16*26]
                                                        +out_temp[16*49*k+16*(28)-1:16*49*k+16*27]
                                                        +out_temp[16*49*k+16*(29)-1:16*49*k+16*28]
                                                        +out_temp[16*49*k+16*(30)-1:16*49*k+16*29]
                                                        +out_temp[16*49*k+16*(31)-1:16*49*k+16*30]
                                                        +out_temp[16*49*k+16*(32)-1:16*49*k+16*31]
                                                        +out_temp[16*49*k+16*(33)-1:16*49*k+16*32]
                                                        +out_temp[16*49*k+16*(34)-1:16*49*k+16*33]
                                                        +out_temp[16*49*k+16*(35)-1:16*49*k+16*34]
                                                        +out_temp[16*49*k+16*(36)-1:16*49*k+16*35]
                                                        +out_temp[16*49*k+16*(37)-1:16*49*k+16*36]
                                                        +out_temp[16*49*k+16*(38)-1:16*49*k+16*37]
                                                        +out_temp[16*49*k+16*(39)-1:16*49*k+16*38]
                                                        +out_temp[16*49*k+16*(40)-1:16*49*k+16*39]
                                                        +out_temp[16*49*k+16*(41)-1:16*49*k+16*40]
                                                        +out_temp[16*49*k+16*(42)-1:16*49*k+16*41]
                                                        +out_temp[16*49*k+16*(43)-1:16*49*k+16*42]
                                                        +out_temp[16*49*k+16*(44)-1:16*49*k+16*43]
                                                        +out_temp[16*49*k+16*(45)-1:16*49*k+16*44]
                                                        +out_temp[16*49*k+16*(46)-1:16*49*k+16*45]
                                                        +out_temp[16*49*k+16*(47)-1:16*49*k+16*46]
                                                        +out_temp[16*49*k+16*(48)-1:16*49*k+16*47]
                                                        +out_temp[16*49*k+16*(49)-1:16*49*k+16*48];    //dont plus bias
        end
endgenerate
//wire signed[16*10-1:0]data_out;
//assign data_out=out;
//wire [10-1:0]s_in;
//wire [10-1:0]s_out;
//wire signed[16*10-1:0]v_old;
//wire signed[16*10-1:0]v_out;
//generate 
//        genvar m;
//        for(m=0;m<10;m=m+1) begin
//            act aaa(clk,data_out[16*(m+1)-1:16*m],s_in[m],s_out[m],v_old[16*(m+1)-1:16*m],v_out[16*(m+1)-1:16*m]);
//        end
//endgenerate
endmodule

module add_v_fc(clk,rst,v_in,addr_r,addr_w,v_out,b,en);
input clk;
input rst;
input signed[16*10-1:0]v_in;
input [4-1:0]addr_r;
input [4-1:0]addr_w;
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
if(addr_r==4'b0000 & addr_w==4'b1111) 
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
v_temp[175 : 160]<=v_in[175 : 160]+v_temp[175 : 160]+b[175 : 160];
v_temp[191 : 176]<=v_in[191 : 176]+v_temp[191 : 176]+b[191 : 176];
v_temp[207 : 192]<=v_in[207 : 192]+v_temp[207 : 192]+b[207 : 192];
v_temp[223 : 208]<=v_in[223 : 208]+v_temp[223 : 208]+b[223 : 208];
v_temp[239 : 224]<=v_in[239 : 224]+v_temp[239 : 224]+b[239 : 224];
v_temp[255 : 240]<=v_in[255 : 240]+v_temp[255 : 240]+b[255 : 240];
en_reg<=1'b1;
end
else if (addr_r==4'b0001 & addr_w==4'b0000)
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
v_temp[175 : 160]<=v_in[175 : 160];
v_temp[191 : 176]<=v_in[191 : 176];
v_temp[207 : 192]<=v_in[207 : 192];
v_temp[223 : 208]<=v_in[223 : 208];
v_temp[239 : 224]<=v_in[239 : 224];
v_temp[255 : 240]<=v_in[255 : 240];
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
v_temp[175 : 160]<=v_in[175 : 160]+v_temp[175 : 160];
v_temp[191 : 176]<=v_in[191 : 176]+v_temp[191 : 176];
v_temp[207 : 192]<=v_in[207 : 192]+v_temp[207 : 192];
v_temp[223 : 208]<=v_in[223 : 208]+v_temp[223 : 208];
v_temp[239 : 224]<=v_in[239 : 224]+v_temp[239 : 224];
v_temp[255 : 240]<=v_in[255 : 240]+v_temp[255 : 240];
end

end


end
endmodule

module act_fc(clk,v_in,s_old,s_out,v_old,v_out);
// clk is clock   1bit   v_in is result of conv  3.13 16bit
// s_old is old spike from BRAM  s_out is new spike to store in BRAM   1bit
// v_old is old v from BRAM   v_out is new v to stroe in BRAM       3.13 16bit
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