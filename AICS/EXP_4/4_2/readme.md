sh run_cpu.sh

cd fppb_to_intpb

python fppb_to_intpb.py udnie_int8.ini

cd ..

sh run_mlu.sh
