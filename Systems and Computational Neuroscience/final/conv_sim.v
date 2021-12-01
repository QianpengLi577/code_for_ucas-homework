`timescale 1ns / 1ns
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/11/23 17:15:39
// Design Name: 
// Module Name: conv_sim
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


module conv_sim(
//x,y
    );
//input x;
//output [14-1:0]y;
reg clk=0;
initial clk=0;
always #5 clk<=~clk;
reg [3-1:0] xmem[0:16*16-1];
reg signed[16-1:0] wmem[0:3*3-1];
reg signed[16-1:0] bmem[0:1];
reg signed[16-1:0] vv_oldmem[0:14*14-1];
reg  s_inmem[0:14*14-1];
//reg [3-1:0] x_fcmem[0:12-1];
//reg signed[16-1:0] w_fcmem[0:12*10-1];
//reg signed[16-1:0] b_fcmem[0:10-1];

//reg  s_oldmem[0:12-1];
//reg signed[16-1:0] v_oldmem[0:12-1];
//reg signed[16-1:0] v_inmem[0:12-1];

initial begin
$readmemb("E:/UCAS/x.txt",xmem);
$readmemb("E:/UCAS/w.txt",wmem);
$readmemb("E:/UCAS/b.txt",bmem);
$readmemb("E:/UCAS/vv_old.txt",vv_oldmem);
$readmemb("E:/UCAS/spike_in.txt",s_inmem);
//$readmemb("E:/UCAS/x_fc.txt",x_fcmem);
//$readmemb("E:/UCAS/w_fc.txt",w_fcmem);
//$readmemb("E:/UCAS/b_fc.txt",b_fcmem);

//$readmemb("E:/UCAS/s_old.txt",s_oldmem);
//$readmemb("E:/UCAS/v_old.txt",v_oldmem);
//$readmemb("E:/UCAS/v_in.txt",v_inmem);
end

wire [16*16*3-1:0]x;
wire signed[3*3*16-1:0]w;
wire signed[16-1:0]b;
wire signed[14*14*16-1:0]out;
//wire signed[16-1:0]out_temp;
wire [3*7*7-1:0] out_p;
wire [14*14-1:0]s_in;
wire [14*14-1:0]s_out;
wire signed[16*14*14-1:0] vv_old;
conv_act_pool c1(
clk,x,w,b,out,out_p,s_in,s_out,vv_old
    );
//assign y = s_out[14-1:0];
//conv3_3 c(clk,{x[68:60],x[38:30],x[8:0]},w,b,out_temp);

//wire [12*3-1:0]x_fc;
//wire signed[12*10*16-1:0]w_fc;
//wire signed[16*10-1:0]b_fc;
//wire signed[10*16-1:0]out_fc;
//fc_act_pool fc1(
//clk,x_fc,w_fc,b_fc,out_fc
//    );

generate 
            genvar i;
            for(i = 0; i < 16*16; i = i + 1) begin
               assign x[3*(i+1)-1:3*i]=xmem[i];
            end
endgenerate 
assign b=bmem[0];
generate 
            genvar j;
            for(j = 0; j < 3*3;j = j + 1) begin
               assign w[16*(j+1)-1:16*j]=wmem[j];
            end
endgenerate 
generate 
            for(j = 0; j < 14*14;j = j + 1) begin
               assign s_in[j]=s_inmem[j];
            end
endgenerate 
generate 
            for(j = 0; j < 14*14;j = j + 1) begin
               assign vv_old[16*(j+1)-1:16*j]=vv_oldmem[j];
            end
endgenerate 
//generate 
//            genvar i_fc;
//            for(i_fc = 0; i_fc < 12; i_fc = i_fc + 1) begin
//               assign x_fc[3*(i_fc+1)-1:3*i_fc]=x_fcmem[i_fc];
//            end
//endgenerate 
//generate 
//            genvar j_fc;
//            for(j_fc = 0; j_fc < 10; j_fc = j_fc + 1) begin
//               assign b_fc[16*(j_fc+1)-1:16*j_fc]=b_fcmem[j_fc];
//            end
//endgenerate 
//generate 
//            genvar k_fc;
//            for(k_fc = 0; k_fc < 12*10;k_fc = k_fc + 1) begin
//               assign w_fc[16*(k_fc+1)-1:16*k_fc]=w_fcmem[k_fc];
//            end
//endgenerate 


//wire [3*16-1:0]out_pool;
//avgpooling2_2 a1(clk,4'b0000,out_pool[2:0]);
//avgpooling2_2 a2(clk,4'b0001,out_pool[5:3]);
//avgpooling2_2 a3(clk,4'b0010,out_pool[8:6]);
//avgpooling2_2 a4(clk,4'b0011,out_pool[11:9]);
//avgpooling2_2 a5(clk,4'b0100,out_pool[14:12]);
//avgpooling2_2 a6(clk,4'b0101,out_pool[17:15]);
//avgpooling2_2 a7(clk,4'b0110,out_pool[20:18]);
//avgpooling2_2 a8(clk,4'b0111,out_pool[23:21]);
//avgpooling2_2 a9(clk,4'b1000,out_pool[26:24]);
//avgpooling2_2 a10(clk,4'b1001,out_pool[29:27]);
//avgpooling2_2 a11(clk,4'b1010,out_pool[32:30]);
//avgpooling2_2 a12(clk,4'b1011,out_pool[35:33]);
//avgpooling2_2 a13(clk,4'b1100,out_pool[38:36]);
//avgpooling2_2 a14(clk,4'b1101,out_pool[41:39]);
//avgpooling2_2 a15(clk,4'b1110,out_pool[44:42]);
//avgpooling2_2 a16(clk,4'b1111,out_pool[47:45]);

//wire signed[16-1:0] v_out[0:12-1];
//wire  s_out[0:12-1];
//generate 
//            genvar s;
//            for(s = 0; s < 12;s = s + 1) begin
//                act cc(clk,v_inmem[s],s_oldmem[s],s_out[s],v_oldmem[s],v_out[s]);
//            end
//endgenerate 

integer file1,file2,file3;
initial begin
file1 = $fopen("E:/UCAS/result_update.txt","w");
file2 = $fopen("E:/UCAS/result_spike.txt","w");
file3 = $fopen("E:/UCAS/result_pooling.txt","w");
# 40
$fwrite(file1,"%b\n",out);
$fwrite(file2,"%b\n",s_out);
$fwrite(file3,"%b\n",out_p);
$fclose(file1);
$fclose(file2);
$fclose(file3);
#40
$stop ;
end

endmodule
