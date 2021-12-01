`timescale 1ns / 1ns
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/11/23 14:46:28
// Design Name: 
// Module Name: conv_act_pool
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
//              kernel 3*3
//              9*9 conv 
//              weight , bais 3.13  16bit
//              input(spike)  1.2  3bit
//              output(conv_result)  3.13  16bit
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module conv_act_pool(
clk,x,w,b,out,out_p,spike_in,spike_out,vv_old
    );
// input should plus s_in  v_old
//output should plus s_out  v_out
//and out shoule be changed as out_pooling
input clk;// the golbal clock

input [16*16*3-1:0]x; // unsigned  the result after pooling  or the source data  1.2 3bit  10*10
// x is assoclated with  0 1 2 3 ....9 ;10 11 12 ....   need load from BRAM by 3bit
input signed[3*3*16-1:0]w;// signed weight data   3.13 16bit   3*3
// w is associated with 0 1 2;3 4 5 ;6 7 8   need load from BRAM by 16bit
input signed[16-1:0]b;// signed bias data 3.13 16bit   1
// when load w , b is load the same time
input [14*14*1-1:0] spike_in;//FROM BRAM  8*8  1bit
output [14*14*1-1:0] spike_out;//TO BRAM  TO POOLING--out_act   8*8 1bit
input signed[16*14*14*1-1:0] vv_old;//FROM BRAM  8*8  3.13 16bit

output signed[14*14*16-1:0]out;// just test sign output--conv result   8*8   
// it should be the data after pooling   
// this out is not true  besides I think spike and updated v should also outout inorder to write,but it will cost long time to write
output [3*7*7-1:0] out_p;
wire [14*14*1-1:0] s_in;//FROM BRAM  8*8  1bit
wire [14*14*1-1:0] s_out;//TO BRAM  TO POOLING--out_act   8*8 1bit
wire signed[16*14*14*1-1:0] v_old;//FROM BRAM  8*8  3.13 16bit
wire signed[16*14*14*1-1:0] v_out;//TO BRAM   8*8  3.13 16bit

assign s_in=spike_in;
assign spike_out=s_out;
assign v_old=vv_old;

wire signed[14*14*16-1:0] data_out;//the result of conv    to act     signed data  3.13 16bit  8*8
wire [1*14*14-1:0] out_act;//activation layer output of spike   1bit   8*8
wire [3*7*7-1:0] out_pooling;//output of pooling  to fc layer or next state of conv
assign out_act = s_out;
assign out=v_out;
assign out_p=out_pooling;
// conv part
generate 
        genvar i,j;
        for(i = 0; i < 14; i = i + 1) begin
            for(j=0; j < 14; j = j + 1)begin
                  conv3_3 c(
                  .clk(clk),
                  .x({ x[3*16*(i+2)+3*(j+3)-1:48*(i+2)+3*j],x[48*(i+1)+3*(j+3)-1:48*(i+1)+3*j] ,x[48*i+3*(j+3)-1:48*i+3*j] }),
                  .w(w),
                  .b(b),
                  .out(data_out[16*(14*i+j+1)-1:16*(14*i+j)]));          
            end
        end
endgenerate  

//act part 
generate 
        genvar m;
        for(m=0;m<14*14;m=m+1) begin
            act aaa(clk,data_out[16*(m+1)-1:16*m],s_in[m],s_out[m],v_old[16*(m+1)-1:16*m],v_out[16*(m+1)-1:16*m]);
        end
endgenerate  

// pooling part
generate 
        genvar i_p,j_p;
        for(i_p = 0; i_p < 7; i_p = i_p + 1) begin
            for(j_p=0; j_p < 7; j_p = j_p + 1)begin
                  avgpooling2_2 a(clk,{out_act[28*i_p+2*(j_p+1)-1 : 28*i_p + 2*j_p ],out_act[14+28*i_p+2*(j_p+1)-1:14+28*i_p+2*j_p]},out_pooling[3*(7*i_p+j_p+1)-1 : 3*(7*i_p+j_p)]);  
            end
        end
endgenerate  


endmodule

module conv3_3(clk,x,w,b,out);

input clk;
input [3*3*3-1:0]x;//unsigned
input signed[3*3*16-1:0]w;
input signed[16-1:0]b;
output signed[16-1:0]out;
wire signed[16-1:0]data_out;
assign out=data_out;
wire signed[16-1:0] mult_out [9-1:0];

generate 
        genvar i;
        for(i = 0; i < 9; i = i + 1) begin
            mul_conv_fc mul_name (
          .A(w[16*(i+1)-1:16*i]),      // input wire [15 : 0] A
          .B(x[3*(i+1)-1:3*i]),      // input wire [2 : 0] B
          .P(mult_out[i])      // output wire [15 : 0] P
        );
        end
endgenerate   

assign data_out=mult_out[0]
                        +mult_out[1]
                        +mult_out[2]
                        +mult_out[3]
                        +mult_out[4]
                        +mult_out[5]
                        +mult_out[6]
                        +mult_out[7]
                        +mult_out[8]
                        +b;
endmodule

module avgpooling2_2(clk,x,out);
//  all are unsigned data
input [4-1:0]x;
input clk;
output [3-1:0]out;
reg [3-1:0] out_temp;
assign out = out_temp;

always @(*)
begin
if (x[0]&x[1]&x[2]&x[3]) out_temp=3'b100;
// 1111
else if ((~x[0])&(~x[1])&(~x[2])&(~x[3])) out_temp=3'b000;
// 0000
else if ((~x[0]&~x[1]&~x[2]&x[3])|
           (~x[0]&~x[1]&x[2]&~x[3])|
           (~x[0]&x[1]&~x[2]&~x[3])| 
           (x[0]&~x[1]&~x[2]&~x[3])) out_temp=3'b001;
// 0001 0010 0100 1000     
else if ((~x[0]&~x[1]&x[2]&x[3])|
          (~x[0]&x[1]&~x[2]&x[3])|
          (x[0]&~x[1]&~x[2]&x[3])| 
          (~x[0]&x[1]&x[2]&~x[3])|
          (x[0]&~x[1]&x[2]&~x[3])|
          (x[0]&x[1]&~x[2]&~x[3])) out_temp=3'b010;
// 0011 0101 1001 0110 1010 1100  
else   out_temp=3'b011;
// ((~x[0]&x[1]&x[2]&x[3])|
//          (x[0]&~x[1]&x[2]&x[3])|
//          (x[0]&x[1]&x[2]&~x[3])|
//          (x[0]&x[1]&~x[2]&x[3])) 
// 0111 1011 1110 1101        
end
endmodule

module act(clk,v_in,s_old,s_out,v_old,v_out);
// clk is clock   1bit   v_in is result of conv  3.13 16bit
// s_old is old spike from BRAM  s_out is new spike to store in BRAM   1bit
// v_old is old v from BRAM   v_out is new v to stroe in BRAM       3.13 16bit
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
//assign v_out=v_in + {16{~s_old}}&(v_old>>>2);
assign v_out=v_in + $signed({16{~s_old}}&({sign,sign,v_old[15:2]}));
assign s_out=s_temp;
always@(*)
if (v_out>=16'sb0001000000000000)   s_temp<=1'b1;
else s_temp<=1'b0;

endmodule