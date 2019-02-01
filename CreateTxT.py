# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:41:49 2019

@author: Administrator
"""
import os
import argparse

parser = argparse.ArgumentParser(description="show some parse for the progess")
parser.add_argument("-in","--file_name",type=str,help="the name of file to read")
parser.add_argument("-out","--save_file_name",type=str,help="the name of file to write")
parse = parser.parse_args()

def calu_10(List):
    return sum(List)/len(List)

def List2Str(List):
    s = ''
    for l in List:
        length = len(str(l))
        s += str(l) + ' '*(5-length+2)
    return s


def read(file_name):
    print('read!')
    str_array = []
    with open(file_name) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            if line[0] == '0':
                line_list = line.split(' ')
                num_list = []
                for l in line_list:
                    if l != '':
                        try:
                            num_list.append(float(l))
                        except ValueError as e:
                            print('could not convert {} to float'.format(l))
                str_array.append(List2Str(num_list)+' '*5 + str(calu_10(num_list)))
            else:
                str_array.append(line)
    return str_array

def write(file_name,str_array):
    print('write!')
    if os.path.exists(file_name):
        raise NameError('file {} is exists,please change a new file name!'.format(file_name))
    with open(file_name,'w') as f:
        for Str in str_array:
            if Str[-1] != '\n':
                Str += '\n'
            f.write(Str)
    print('save file to ' + os.getcwd()+'\\' + file_name)

def main():
    if parse.file_name is None:
        file_name = ['Network_loss1_1_2.txt','Network_loss2_1_2.txt','Network_loss3_1_2.txt','Network_loss4_1_2.txt']
        for f in file_name:
            str_array = read(f)
            if parse.save_file_name is None:
                parse.save_file_name = f.split('.')[0] + '_result.txt'
            write(parse.save_file_name,str_array)
    else:
        str_array = read(parse.file_name)
        if parse.save_file_name is None:
            parse.save_file_name = parse.file_name.split('.')[0] + '_result.txt'
        write(parse.save_file_name,str_array)
        
if __name__ == "__main__":
#    s = './CTG/Network_loss4_1_1.txt'
#    a = read(s)
#    write('aaaa.txt',a)
    main()











    
