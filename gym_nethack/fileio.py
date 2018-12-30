import re, os

DIR_CHAR = '\\' if os.name == 'nt' else '/'

def append(arg, file, mode="a+"):
    with open(file+".txt", mode) as rfile:
        rfile.write(str(arg) + "\n")

def read_line_list(file, ignore=[], load_float=True, add_txt=True):
    file = file + ".txt" if not ".txt" in file and add_txt else file
    if not os.path.isfile(file):
        print("Incorrect path:", file)
        return []
    f = open(file, 'r')
    lst = []
    for line in f:
        if f not in ignore:
            lst.append(float(line) if load_float else line)
    f.close()
    return lst

def get_dir_for_params(params, abbreviations):
    param_dir = ''
    for i, (param, param_str) in enumerate(zip(params, abbreviations)):
        param_dir += param_str
        if type(param) is bool:
            param_dir += str(int(param))
        else: param_dir += str(param)
        if i < len(abbreviations)-1:
            param_dir += '_'
    param_dir += '/'
    return param_dir