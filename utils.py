# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:23:08 2020

@author: janeiroja
"""

import datetime as dt
import os
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt


# import time
import pickle


# Some modules are quite optional, since they are only used for some features
# In these cases, we import with a protective import: only imports if it exists
try:
    import PyPDF2 # Needs 'pdf' package installed.
except ModuleNotFoundError:
    # Error handling
    print('Could not find PyPDF module. Did you pip it into the current env?')
    print('Merge of PDF files will not work...')
    


"""
Parsing functions, to check if the input to a function/(...)/class is within
what that function/(...)/class expects to get
"""


def parse_input(arg, values):
    if arg not in values:
        raise Exception(f"Wrong input value! Must have a value from {values}. "
                        f"Instead, the input was {arg}.")


def parse_input_type(arg, types):
    """
    types is a list of types.
    E.g. to check if the input is either an int or a float:
        parse_input_type(arg, ['int', 'float']):
    """
    if type(arg).__name__ not in types:
        raise Exception(f"Wrong input type! Must have a type from {types}.\n"
                        f"The type was instead {type(arg).__name__}.")


def parse_input_lists(arg, value_types, length=-1):
    # 
    
    if length != -1 and len(arg) != length:
        raise Exception(f"List with wrong amount of elements. It should have "
                        f"{length} elements")
    
    if len(arg) > 0:
        if type(arg[0]).__name__ not in value_types:
            raise Exception(f"Wrong input type! "
                            f"Must have a type from {value_types}")
        

def parse_input_lists_intensive(arg, value_types, length=-1):
    # Check if list has the correct amount of values and value types
    # Checks all values for their type, unless it's a nd_array.
    
    if length != -1 and len(arg) != length:
        raise Exception(f"List with wrong amount of elements. It should have "
                        f"{length} elements")
    
    if len(arg) > 0:
        for ele in arg:
            if type(ele).__name__ not in value_types:
                raise Exception(f"Wrong input type! "
                                f"Must have a type from {value_types}")


""" Time and Timestamp related functions """


def timestamp(s=0, ms=0, us=0):
    # Every time instant will be a datetime.timedelta object
    # Seconds are obligatory, others go by order
    
    if type(s).__name__ == 'list' or type(s).__name__ == 'numpy.ndarray':
        print('At the moment, there is no support '
              'for timestamping a list of instants.')
        pass
    
    parse_input_type(s, ['int', 'float', 'float64'])
    parse_input_type(ms, ['int', 'float', 'float64'])
    parse_input_type(us, ['int', 'float', 'float64'])
    
    return dt.timedelta(seconds=s, milliseconds=ms, microseconds=us)


def print_timestamp(t, end_str='\n'):
    """
    Prints a datetime.timedelta (timestamp) in a more readable way.
    """
    
    parse_input_type(t, 'datetime.timedelta')
    
    print(f"{t.__str__()[5:]}s", end=end_str)
    

def get_seconds(t):
    """
    Returns how many seconds there are in the timestamp. Converts from
    microseconds!
    """
    
    return t.microseconds / 1e6 + t.seconds


def get_time():
    return dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")



"""
Functions for simple operations with numbers.
"""


def plot_complex(c):
    if type(c).__name__ == 'complex128':
        plt.scatter(c.real, c.imag)
    elif type(c).__name__ == 'list' and len(c) != 0:
        parse_input_type(c[0], ['complex128'])
        
        for cn in c:
            plt.scatter(cn.real, cn.imag)
    else:
        raise Exception('Only complex or lists of complex numbers')
        
    plt.grid()
    plt.show()


def round_to_value(number, roundto):
    """
    Rounds a number to the nearest... 0.5. 92.3 -> 92.5.
    From https://stackoverflow.com/questions/4265546/, answer from Dave Webb.
    """
    
    if number % (roundto / 2) == 0:
        # This means it's exactly in the middle! Use ceil instead!
        number += roundto / 10
    
    return (round(number / roundto) * roundto)


def success_coin_flip(prob_of_error):
    """ Returns True with chance (1 - prob_of_error). False otherwise."""
    return np.random.uniform() > prob_of_error


def divisors(n):
    div_list = []
    for i in range(1, int(np.ceil(np.sqrt(n)))):
        if n % i == 0:
            div_list.append(i)
    
    remaining_divisors = []
    for divisor in div_list:
        remaining_divisors.append(int(n / divisor))
    
    if np.sqrt(n) == int(np.sqrt(n)):
        div_list += [int(np.sqrt(n))]
    
    div_list += sorted(remaining_divisors)
    return div_list


def non_obvious_divisors(n):
    """ Divisors without 1 and n, those are obvious."""
    
    div_list = divisors(n)
    return div_list[1:-1]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


"""
IO functions:
    - Save & Load var picle (using Pickle)
    - Save statistics to CSV file (using NumPy) for quick Excel analysis

"""


def makedirs(path):
    """
    Wrapper for os.makedirs function.
    
    The only purpose of this sort of function is to allow a better
    organisation of the imports.
    """
    return os.makedirs(path)


def isdir(path):
    """
    Wrapper for os.path.isdir
    """    
    return os.path.isdir(path)


def del_dir(path):
    """
    Deletes directory/folder and all contents in it.
    """
    shutil.rmtree(path)
    
    
def del_file(path):
    """
    Deletes file.
    """
    os.remove(path)
    
    
def get_computer_name():
    return os.environ['COMPUTERNAME']    
    
def get_var_name(var, globals_dict):
    """
    Returns one of the occurrences of the variable value in the environment.
    If there's only one variable with that name, returns the correct name of
    the variable.
    
    There's a little twist for not returning anything started with '_', which
    seems to yield better results.
    """
    for var_name in globals_dict:
        if globals_dict[var_name] is var:
            if var_name[0] == '_':
                continue
            return var_name


def save_var_pickle(var, directory='', globals_dict={}):
    """
    Saves variable with the variable name to the directory.
    """
    fname = get_var_name(var, globals_dict)
        
    pickle.dump(var, open(directory + f"{fname}.p", "wb"))


def load_var_pickle(var_name, directory):
        
    return pickle.load(open(directory + f"{var_name}.p", "rb"))







""" 
Other Functions: 
    - cpu count
    - find last occurrence of an item in a list
    - find divisors of a number
    - Make python lists
"""


def get_cpu_count():
    return os.cpu_count()


def get_cwd():
    return os.getcwd()

def stop_execution():
    sys.exit()


def get_input_arg(arg_idx):
    """
    arg_idx 0 is the name of the script.
    arg_idx 1 is the first argument
    arg_idx 2 is the second argument
    ...
    arg_idx -1 is the LAST argument
    arg_idx -2 is the second from argument
    ...
    """
    if abs(arg_idx) >= len(sys.argv):
        return ''
        # raise Exception('There are not that many command line arguments.')
    else:
        return sys.argv[arg_idx]


def is_imported(module_name):
    return module_name in sys.modules



def find_string_in_files(s, file_list):
    
    first_match = True
    
    for file in file_list:
        try:
            with open(file) as fp:
                if s in fp.read():
                    if first_match:
                        first_match = False
                        print('String found in files:')
                    print(file)
        except:
            print("Doesn't exist: " + file)
            continue
        
                
def find_out_of_memory(inst_folder):
    files_list = [inst_folder + f"\Instance{i}\log.txt" for i in range(200)]
    find_string_in_files('Out of memory', files_list)


def find_last_occurence(target_list, item):
    """
    Return the index (on the original list) of the last item with the same
    value as the item.
    """
    for i in reversed(range(len(target_list))):
        if target_list[i] == item:
            return i
    else:
        raise ValueError(f"{item} is not in the target list") 


def first_nonzero(arr, axis, invalid_val=-1):
    """
    Finds index of first non-zero value
    From: https://stackoverflow.com/questions/47269390
    """
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)



def make_py_list(dim, siz):
    """
    Create a python list of the size specified.
    siz should be an array, one entry per dimension.
    If the size length is smaller than the dimensions, the list will be a list
    of empty lists.
    """
    
    if len(siz) > dim:
        raise Exception('More dimensions given than the maximum dimension!')
    if len(siz) == dim:
        l = np.zeros(siz).tolist()
        return l
    else:
        new_size = siz + [1]
        l = np.zeros(new_size).tolist()
        
        if dim == 2:
            for idx in range(new_size[0]):
                l[idx] = []
        
        if dim == 3:
            for idx0 in range(new_size[0]):
                for idx1 in range(new_size[1]):
                    l[idx0][idx1] = []
                    
        if dim == 4:
            for idx0 in range(new_size[0]):
                for idx1 in range(new_size[1]):
                    for idx2 in range(new_size[2]):
                        l[idx0][idx1][idx2] = []
            
        if dim > 4:
            raise Exception('Make Py list only implements for dim <= 4')
            
        return l

def elementwise_comparison_loop(l1, l2):
    for i in range(min(len(l1), len(l2))):
        if l1[i] != l2[i]:
            return False
    return len(l1) == len(l2)

"""
For Fun:
    - Line count n' print functions
"""

def rawcount(filename):
    # From Michael Bacon, in:
    # https://stackoverflow.com/questions/845058/
    # There are some crazy efficient things out there...
    # This is a good trade-off between complexity and performance.
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines


def flex(percentages=False):
    
    working_dir = get_cwd()
    
    # MATLAB
    matlab_main_func_path = working_dir + '\\Matlab\\Meeting12.m'
    
    matlab_aux_func_dir = working_dir + '\\Matlab\\Aux Functions'
    
    matlab_aux_script_dir = working_dir + '\\Matlab\\Aux Scripts'
    
    aux_funcs_files = list(os.walk(matlab_aux_func_dir))[0][2]
    aux_scripts_files = list(os.walk(matlab_aux_script_dir))[0][2]
    
    matlab_main_count = rawcount(matlab_main_func_path)
    matlab_aux_func_total = sum(rawcount(f"{matlab_aux_func_dir}\\{f}") 
                                for f in aux_funcs_files)
    matlab_aux_script_total = \
        sum(rawcount(f"{matlab_aux_script_dir}\\{f}") 
            for f in aux_scripts_files)
    
    matlab_aux_total_sum = matlab_aux_func_total + matlab_aux_script_total
    
    # PYTHON
    python_main_func_path1 = working_dir + '\\sxr_sim.py'
    python_main_func_path2 = working_dir + '\\sxr_gen.py'
    
    python_aux_func_dir = working_dir
    
    
    files = list(os.walk(python_aux_func_dir))[0][2]
    
    python_main_count = (rawcount(python_main_func_path1) + 
                         rawcount(python_main_func_path2))
    python_aux_total_sum = sum([rawcount(f"{python_aux_func_dir}\\{f}") 
                                for f in files if f[-3:] == '.py'])
    python_aux_total_sum -= python_main_count
    
    
    
    python_total = python_main_count + python_aux_total_sum
    matlab_total = matlab_main_count + matlab_aux_total_sum
    
    tot_sum = python_total + matlab_total
    

    if percentages:    
        import prettytable
        
        py_main_percent = round(python_main_count / python_total * 100)
        py_aux_percent = round(python_aux_total_sum / python_total * 100)
    
        mat_main_percent = round(matlab_main_count / matlab_total * 100)
        mat_aux_percent = round(matlab_aux_total_sum / matlab_total * 100)
        
        py_percent = round(python_total / tot_sum * 100)
        mat_percent = round(matlab_total / tot_sum * 100)
        
        t1 = prettytable.PrettyTable([f"{'':5}", 'Matlab', 'Python'])
        t1.add_row(['Main', 
                    f"{matlab_main_count} ({mat_main_percent}%)",
                    f"{python_main_count} ({py_main_percent}%)"])
        t1.add_row(['Aux', 
                    f"{matlab_aux_total_sum} ({mat_aux_percent}%)",
                    f"{python_aux_total_sum} ({py_aux_percent}%)"])
        
        
        t2 = prettytable.PrettyTable()
        t2.header = False
        t2.add_row(['Total', 
                    f"{matlab_total} ({mat_percent}%)",
                    f"{python_total} ({py_percent}%)",
                    f"{tot_sum}"])
        
        print(t1)
        print(t2)
    else:
        print(f"Matlab:\n"
              f"Lines in main: {matlab_main_count}\n"
              f"Lines in auxs: {matlab_aux_total_sum}")
    

        print(f"Python:\n"
              f"Lines in main: {python_main_count}\n"
              f"Lines in auxs: {python_aux_total_sum}")
    
        print(f"Total Sum: {tot_sum}")





"""
Tools to facilitate parameter choices
"""


def get_max_compression(numerologies):
    return 2 ** (max(numerologies) - min(numerologies))


def get_possible_time_divisions(sim_ttis, max_compression):
    div_list = divisors(sim_ttis)
    return [i for i in div_list if sim_ttis / i % max_compression == 0]


"""
Help with Matplotlib.
"""

def figure_pos(x=0, y=0, dx=0, dy=0):
    # Get the current manager to move figures to other screens
    mngr = plt.get_current_fig_manager()
    
    if x == 0 and y == 0 and dx == 0 and dy == 0:
        # In case knowing x, y, dx, dy is needed: get the QTCore PyRect object
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
    else:
        mngr.window.setGeometry(x, y, dx, dy)
        
        
        
        
        
        
"""
PDF concat feature
"""

def pdf_cat(input_files, output_stream):
    input_streams = []
    try:
        # First open all the files, then produce the output file, and
        # finally close the input files. This is necessary because
        # the data isn't read from the input files until the write
        # operation. Thanks to
        # https://stackoverflow.com/questions/6773631/
        for input_file in input_files:
            input_streams.append(open(input_file, 'rb'))
        writer = PyPDF2.PdfFileWriter()
        for reader in map(PyPDF2.PdfFileReader, input_streams):
            for n in range(reader.getNumPages()):
                writer.addPage(reader.getPage(n))
        writer.write(output_stream)
    finally:
        for f in input_streams:
            f.close()
            

"""
Other tools
"""
            
def get_all_files_of_format(folder, the_format=''):
    
    all_files = list(os.walk(folder))[0][2]
    
    if the_format == '':
        # Then all files should be considered. 
        # Apply an 'allow all' filter
        filter_func = lambda x: (True)
    else: # a filter for that format needs to be applied
        if the_format[0] != '.':
            the_format = '.' + the_format
    
        filter_func = lambda x: (x[-4:] == the_format)
    
    files_with_format = list(filter(filter_func, all_files))
    
    return files_with_format
