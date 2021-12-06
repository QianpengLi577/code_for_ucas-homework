`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2021/12/01 15:13:48
// Design Name: 
// Module Name: snn
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


module snn(
input clk, //clk
input rst, //reset 
output [10-1:0]spike  //spike
    );
wire en_load; //2cycle high level to load data   3 cycle low level to count
    
wire data_1; // check the data of first simulation time 
wire  [6-1:0]addr_r; //read addr
wire [6-1:0]addr_w; //write addr
wire  flag;//one img finsh signal
wire en_b;//enable signal of addication bias
assign en_b = addr_r ==6'd27 ;
wire en_w;// write en
assign en_w = !(addr_r===addr_w);
generate_state a1(clk,en_load,rst&(!flag));  //generate en_load signal
generate_add a2(en_load,addr_r,addr_w,flag,rst&(!flag),data_1);//generate addr_r addr_w
wire [28-1:0]x; // the row of img   1 bit
wire [16*28*10-1:0]w;//weight   16bit  1 sign bit  4 int bit  11 float bit
wire [10*16-1:0]b;// bias 16bit same to weight
wire [16*28*10-1:0]v_w;//w*x  the output of synapse
wire [16*10-1:0] v_in;//the sum of v_w

//2021 12 1 16:33  todo   list  :
// generate the synapse and neuron
// generate the BRAM to stroe bias and weight
// write the python code to generate bias and weight .coe file

//generate synapse part
generate 
    genvar i,j;
    for(i=0;i<10;i=i+1)begin
        for(j=0;j<28;j=j+1)begin
            synapse sp(x[j], w[16*28*i+16*(j+1)-1:16*28*i+16*j], v_w[16*28*i+16*(j+1)-1:16*28*i+16*j]);
        end
    end
endgenerate
// to get the input v of  10 neuron
generate 
    genvar m;
    for(m=0;m<10;m=m+1)begin
       assign v_in[16*(m+1)-1:16*m]=   v_w[16*28*m+16*1-1:16*28*m+16*0]
    +v_w[16*28*m+16*2-1:16*28*m+16*1]
    +v_w[16*28*m+16*3-1:16*28*m+16*2]
    +v_w[16*28*m+16*4-1:16*28*m+16*3]
    +v_w[16*28*m+16*5-1:16*28*m+16*4]
    +v_w[16*28*m+16*6-1:16*28*m+16*5]
    +v_w[16*28*m+16*7-1:16*28*m+16*6]
    +v_w[16*28*m+16*8-1:16*28*m+16*7]
    +v_w[16*28*m+16*9-1:16*28*m+16*8]
    +v_w[16*28*m+16*10-1:16*28*m+16*9]
    +v_w[16*28*m+16*11-1:16*28*m+16*10]
    +v_w[16*28*m+16*12-1:16*28*m+16*11]
    +v_w[16*28*m+16*13-1:16*28*m+16*12]
    +v_w[16*28*m+16*14-1:16*28*m+16*13]
    +v_w[16*28*m+16*15-1:16*28*m+16*14]
    +v_w[16*28*m+16*16-1:16*28*m+16*15]
    +v_w[16*28*m+16*17-1:16*28*m+16*16]
    +v_w[16*28*m+16*18-1:16*28*m+16*17]
    +v_w[16*28*m+16*19-1:16*28*m+16*18]
    +v_w[16*28*m+16*20-1:16*28*m+16*19]
    +v_w[16*28*m+16*21-1:16*28*m+16*20]
    +v_w[16*28*m+16*22-1:16*28*m+16*21]
    +v_w[16*28*m+16*23-1:16*28*m+16*22]
    +v_w[16*28*m+16*24-1:16*28*m+16*23]
    +v_w[16*28*m+16*25-1:16*28*m+16*24]
    +v_w[16*28*m+16*26-1:16*28*m+16*25]
    +v_w[16*28*m+16*27-1:16*28*m+16*26]
    +v_w[16*28*m+16*28-1:16*28*m+16*27];

    end
endgenerate

//fc part 


wire [16*10-1:0]v_fc_sum;//every time the sum of v_in
wire en_fc;//neuron can update v and fire spike?
wire [16*10-1:0]v_fc_old;//last time v of neuron
wire [16*10-1:0]v_fc_out;//updated v of neuron
wire [10-1:0]s_fc_old;//last time spike
wire [10-1:0]s_fc_out;//fired spike
wire [10-1:0]s_fc_old_true;// v & data_1   this singal is used to reflash spike(spike of first time shoule be zero)
wire [16*10-1:0]v_fc_old_true;// silmar to s_fc_old_true
assign s_fc_old_true=s_fc_old&{10{data_1}};
assign v_fc_old_true=v_fc_old&{160{data_1}};

add_v_fc FC111(en_load,rst&(!flag),v_in,addr_r,addr_w,v_fc_sum,b,en_fc);//add
neuron_act_fc fccc(clk,v_fc_sum,s_fc_old_true,s_fc_out,v_fc_old_true,v_fc_out);//activation function

assign spike={16{!en_load&en_fc}}&s_fc_out;

//SPIKE
BRAM_FC_SPIKE S (
  .clka(clk),            // input wire clka
.ena(en_fc&!en_load&en_w | flag),              // input wire ena
.wea(en_fc&!en_load&en_w |flag),              // input wire [0 : 0] wea
.addra(addr_w&{6{!flag}}),          // input wire [0 : 0] addra
.dina(s_fc_out&{10{!flag}}),            // input wire [9 : 0] dina
.clkb(clk),          // input wire clkb
 .enb(en_load),              // input wire enb
 .addrb(addr_r),          // input wire [0 : 0] addrb
 .doutb(s_fc_old)         //
);
//V
BRAM_FC_V your_instance_name (
  .clka(clk),            // input wire clka
.ena((en_fc&!en_load&en_w) | flag),              // input wire ena
.wea((en_fc&!en_load&en_w) | flag),              // input wire [0 : 0] wea
.addra(addr_w&{6{!flag}}),          // input wire [0 : 0] addra
.dina(v_fc_out&{160{!flag}}),            // output wire [159 : 0] douta
.clkb(clk),      // input wire clkb
  .enb(en_load),           // input wire [0 : 0] web
.addrb(addr_r),                // input wire [159 : 0] dinb
.doutb(v_fc_old)          
);

// BRAM  part  to load bias  weight  and  x
wire fc_b_busy,fc_w_be_busy,fc_w_ah_busy,x_busy;
BRAM_FC_B load_fc_b (
  .clka(clk),            // input wire clka
  .rsta(!rst),            // input wire rsta
  .ena(en_load),              // input wire ena
  .wea(1'b0),              // input wire [0 : 0] wea
  .addra(1'b0),          // input wire [0 : 0] addra
  .dina(b),            // input wire [159 : 0] dina
  .douta(b),          // output wire [159 : 0] douta
  .rsta_busy(fc_b_busy)  // output wire rsta_busy
);

BRAM_FC_W_AHEAD load_fc_w_ah (
  .clka(clk),            // input wire clka
  .rsta(!rst),            // input wire rsta
  .ena(en_load),              // input wire ena
  .wea(1'b0),              // input wire [0 : 0] wea
  .addra(addr_r),          // input wire [4 : 0] addra
  .dina(w[2240-1:0]),            // input wire [2239 : 0] dina
  .douta(w[2240-1:0]),          // output wire [2239 : 0] douta
  .rsta_busy(fc_w_ah_busy)  // output wire rsta_busy
);

BRAM_FC_W_BEHIND load_fc_w_be (
  .clka(clk),            // input wire clka
  .rsta(!rst),            // input wire rsta
  .ena(en_load),              // input wire ena
  .wea(1'b0),              // input wire [0 : 0] wea
  .addra(addr_r),          // input wire [4 : 0] addra
  .dina(w[4480-1:2240]),            // input wire [2239 : 0] dina
  .douta(w[4480-1:2240]),          // output wire [2239 : 0] douta
  .rsta_busy(fc_w_be_busy)  // output wire rsta_busy
);

BRAM_X load_x (
  .clka(clk),            // input wire clka
  .rsta(!rst),            // input wire rsta
  .ena(en_load),              // input wire ena
  .wea(1'b0),              // input wire [0 : 0] wea
  .addra(addr_r),          // input wire [4 : 0] addra
  .dina(x),            // input wire [27 : 0] dina
  .douta(x),          // output wire [27 : 0] douta
  .rsta_busy(x_busy)  // output wire rsta_busy
);
endmodule





