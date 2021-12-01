`timescale 1ns / 1ns
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/11/29 14:49:35
// Design Name: 
// Module Name: loadbram_sim
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


module loadbram_sim(

    );


reg clk=0;
initial clk=0;
always #5 clk<=~clk;
reg rst=1;
wire en_load;
wire [4-1:0]addr_w;
wire [4-1:0]addr_r;
wire flag;
wire signed[16*3*3-1:0]conv_w;
wire signed[16*1-1:0]conv_b;
wire data_1_flag;
//wire pg;
//wire rsta_busy;
//wire rstb_busy;
generate_state a1(clk,en_load,rst&(!flag));
generate_add a2(en_load,addr_w,addr_r,flag,rst&(!flag),data_1_flag);
BRAM_CONV_W b1 (
  .clka(clk),    // input wire clka
  .ena(en_load),      // input wire ena
  .wea(1'b0),      // input wire [0 : 0] wea
  .addra(addr_r),  // input wire [3 : 0] addra
  .dina(conv_w),    // input wire [143 : 0] dina
  .douta(conv_w)  // output wire [143 : 0] douta
);
BRAM_CONV_B b2 (
  .clka(clk),    // input wire clka
  .ena(en_load),      // input wire ena
  .wea(1'b0),      // input wire [0 : 0] wea
  .addra(addr_r),  // input wire [3 : 0] addra
  .dina(conv_b),    // input wire [15 : 0] dina
  .douta(conv_b)  // output wire [15 : 0] douta
);

wire [16*16*3-1:0]x; //
wire signed[14*14*16-1:0]out;
//wire signed[16-1:0]out_temp;
wire [3*7*7-1:0] out_p;
wire [14*14-1:0]s_in;
wire [14*14-1:0]s_out;
wire signed[16*14*14-1:0] vv_old;
wire [14*14-1:0]s_true;
wire [16*14*14-1:0]v_old_true;
assign s_true=s_in&{196{data_1_flag}};
assign v_old_true=vv_old&{3136{data_1_flag}};
conv_act_pool c1(
clk,x,conv_w,conv_b,out,out_p,s_true,s_out,v_old_true
    );
wire  signed[7*7*10*16-1:0]w_fc;
wire  signed[10*16-1:0]out_fc;
fc_act_pool das(    clk,out_p,w_fc,out_fc        );    
//clk,x,conv_w,conv_b,out,out_p,s_in,s_out,vv_old
//    );
BRAM_FC_W w_fc1 (
  .clka(clk),    // input wire clka
.ena(en_load),      // input wire ena
.wea(1'b0),      // input wire [0 : 0] wea
.addra(addr_r),  // input wire [3 : 0] addra
.dina(w_fc[3919 : 0]),    // input wire [143 : 0] dina
.douta(w_fc[3919 : 0])   // output wire [3919 : 0] douta
);
BRAM_FC_W_BELOW w_fc2 (
  .clka(clk),    // input wire clka
.ena(en_load),      // input wire ena
.wea(1'b0),      // input wire [0 : 0] wea
.addra(addr_r),  // input wire [3 : 0] addra
.dina(w_fc[7839 : 3920]),    // input wire [143 : 0] dina
.douta(w_fc[7839 : 3920]) // output wire [3919 : 0] douta
);
wire en_w ;
assign en_w = !(addr_w===addr_r);   
wire signed[14*14*16-1:0]out_temp;
wire rsta_busy_spike;
wire rstb_busy_spike;
wire rsta_busy_vold;
wire rstb_busy_vold;
assign out_temp = out;

BRAM_CONV_SPIKE your(
  .clka(clk),    // input wire clka
  .ena(en_load&en_w),      // input wire ena
  .wea(en_load&en_w),      // input wire [0 : 0] wea
  .addra(addr_w),  // input wire [3 : 0] addra
  .dina(s_out),    // input wire [195 : 0] dina
  .clkb(clk),    // input wire clkb
  .enb(en_load),      // input wire enb
  .addrb(addr_r),  // input wire [3 : 0] addrb
  .doutb(s_in)  ,
   .rstb((!rst)|(flag)),
   .rsta_busy(rsta_busy_spike),  // output wire rsta_busy
     .rstb_busy(rstb_busy_spike)  // output wire [195 : 0] doutb
);

BRAM_CONV_VOLD  yp (
  .clka(clk),            // input wire clka
  .ena(en_load&en_w),              // input wire ena
  .wea(en_w&en_load),              // input wire [0 : 0] wea
  .addra(addr_w),          // input wire [3 : 0] addra
  .dina(out_temp),            // input wire [3135 : 0] dina
  .clkb(clk),            // input wire rstb
  .enb(en_load),              // input wire enb
  .addrb(addr_r),          // input wire [3 : 0] addrb
  .doutb(vv_old)  ,
             // input wire clkb
  .rstb((!rst)|(flag)),            // input wire rstb
        // output wire [3135 : 0] doutb
  .rsta_busy(rsta_busy_vold),  // output wire rsta_busy
  .rstb_busy(rstb_busy_vold)      // output wire [3135 : 0] doutb
);

blk_mem_gen_0 your_instance_name (
  .clka(clk),    // input wire clka
  .ena(en_load),      // input wire ena
  .wea(1'b0),      // input wire [0 : 0] wea
  .addra(addr_r[1:0]),  // input wire [1 : 0] addra
  .dina(x),    // input wire [767 : 0] dina
  .douta(x)  // output wire [767 : 0] douta
);

wire [16*10-1:0]v_fc_sum;
wire en_fc;
wire [16*10-1:0]b_fc;
wire [16*10-1:0]v_fc_old;
wire [16*10-1:0]v_fc_out;
wire [10-1:0]s_fc_old;
wire [10-1:0]s_fc_out;
wire [10-1:0]s_fc_old_true;
wire [16*10-1:0]v_fc_old_true;
assign s_fc_old_true=s_fc_old&{10{data_1_flag}};
assign v_fc_old_true=v_fc_old&{160{data_1_flag}};
add_v_fc FC111(en_load,rst&(!flag),out_fc,addr_r,addr_w,v_fc_sum,b_fc,en_fc);
act_fc fccc(clk,v_fc_sum,s_fc_old_true,s_fc_out,v_fc_old_true,v_fc_out);
wire rsta_busy_fc_s;
wire rstb_busy_fc_s;
BRAM_FC_SPIKE S_ (
  .clka(clk),            // input wire clka
  .ena(en_fc&!en_load&en_w | flag),              // input wire ena
  .wea(en_fc&!en_load&en_w |flag),              // input wire [0 : 0] wea
  .addra(addr_w&{4{!flag}}),          // input wire [0 : 0] addra
  .dina(s_fc_out&{160{!flag}}),            // input wire [9 : 0] dina
  .clkb(clk),            // input wire clkb
  .rstb(!rst),            // input wire rstb
  .enb(en_load),              // input wire enb
  .addrb(addr_r),          // input wire [0 : 0] addrb
  .doutb(s_fc_old),          // output wire [9 : 0] doutb
  .rsta_busy(rsta_busy_fc_s),  // output wire rsta_busy
  .rstb_busy(rstb_busy_fc_s)  // output wire rstb_busy
);

BRAM_FC_B BB (
  .clka(clk),    // input wire clka
  .ena(en_load),      // input wire ena
  .wea(1'b0),      // input wire [0 : 0] wea
  .addra(1'b0),  // input wire [0 : 0] addra
  .dina(b_fc),    // input wire [159 : 0] dina
  .douta(b_fc)  // output wire [159 : 0] douta
);

wire rsta_busy_fc_v;
wire rstb_busy_fc_v;
BRAM_FC_VOLD aasd (
  .clka(clk),            // input wire clka
  .ena((en_fc&!en_load&en_w) | flag),              // input wire ena
  .wea((en_fc&!en_load&en_w) | flag),              // input wire [0 : 0] wea
  .addra(addr_w&{4{!flag}}),          // input wire [0 : 0] addra
  .dina(v_fc_out&{160{!flag}}),            // output wire [159 : 0] douta
  .clkb(clk),            // input wire clkb
  .rstb(!rst),            // input wire rstb
  .enb(en_load),           // input wire [0 : 0] web
  .addrb(addr_r),                // input wire [159 : 0] dinb
  .doutb(v_fc_old),          // output wire [159 : 0] doutb
  .rsta_busy(rsta_busy_fc_v),  // output wire rsta_busy
  .rstb_busy(rstb_busy_fc_v)  // output wire rstb_busy
);

//assign y=out_p;
integer file1,file2,file3,file4;
initial begin
file1 = $fopen("E:/UCAS/result_1130.txt","w");
file2 = $fopen("E:/UCAS/result_1130_1.txt","w");
file3 = $fopen("E:/UCAS/result_1130_fc.txt","w");
#50
rst=0;
#150
rst=1;
#40
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
#80
$fwrite(file1,"%b\n",out_p);
$fwrite(file3,"%b\n",out_fc);
$fclose(file1);
$fclose(file3);
# 23220
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
#80
$fwrite(file2,"%b\n",out_p);
$fclose(file2);
#10000
$stop ;
end
initial begin
file4 = $fopen("E:/UCAS/result_1201.txt","w");
#1520
$fwrite(file4,"%b\n",v_fc_out);
#1280
$fwrite(file4,"%b\n",v_fc_out);
#1280
$fwrite(file4,"%b\n",v_fc_out);
$fclose(file4);
end
endmodule
