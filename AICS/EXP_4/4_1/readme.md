sh run_cpu.sh
cd fppb_to_intpb
python fppb_to_intpb.py vgg19_int8.ini
sh run_mlu.sh


vgg19_int8.pb is too big to upload
