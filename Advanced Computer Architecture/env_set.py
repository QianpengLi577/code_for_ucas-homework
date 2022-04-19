import sys
import os
import shutil
import argparse
from glob import *

# Parse and validate arguments
# ==============================================================================
parser = argparse.ArgumentParser(
    description='For generating cfg file ')
parser.add_argument('--design', '-d',
                    default='file_test', help='Design name which is same to you top file name ')

# copy cfg file
# ==============================================================================
def mycopyfile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)
        # print("copy %s -> %s" % (srcfile, dstpath + fname))

args = parser.parse_args()
proj_path = os.path.abspath('..')
src_dir = proj_path + '/scripts/gcd/'
dst_dir = proj_path + '/scripts/' + args.design + '/'
if (not os.path.exists(dst_dir)):
    os.mkdir(dst_dir)
src_file_list = glob(src_dir + '*')
for srcfile in src_file_list:
    mycopyfile(srcfile, dst_dir)

path_rtl = proj_path + '/rtl/' + args.design + '/'

# get rtl file name
# ==============================================================================
def getFileName(path):
    f_list = os.listdir(path)
    file_name = []
    for i in f_list:
        if os.path.splitext(i)[1] == '.v':
            file_name.append(i)
    return file_name


file_name = getFileName(path_rtl)
# print(file_name)
file_wr=''
for i in range(len(file_name)):
    if (i == len(file_name)-1):
        file_wr = file_wr + '$RTL_PATH/' + file_name[i] + ' \\'
    else:
        file_wr = file_wr + '$RTL_PATH/' + file_name[i] + ' \\\n'
# print(file_wr)

# generate synth.yosys_0.9.tcl
# ==============================================================================
f1 = open(dst_dir+'synth.yosys_0.9.tcl', 'r+', encoding='utf-8')
content = f1.read()
f1.close()
customary_content = '$RTL_PATH/gcd.v \\'
name = content.replace(customary_content, file_wr)
if customary_content not in content:
    print("source file dont have scripts:"+customary_content)
else:
    with open(dst_dir+'synth.yosys_0.9.tcl', "w", encoding='utf-8') as f2:
        f2.write(name)
        print("generate synth.yosys_0.9.tcl successfully!")

# generate flow_cfg.py
# ==============================================================================
with open(proj_path + '/scripts/cfg/'+'flow_cfg.py', 'a', encoding='utf-8') as f:
    f.write('\n'+args.design+"      = Flow(\'"+args.design+"\',\'sky130\',\'HS\',\'TYP\')")
f.close()
