# Matlab Interface Functions & Wrappers

import subprocess
from joblib import Parallel, delayed


import scipy.io
import h5py

import utils as ut


import numpy as np

import os
import shutil 

import time
import random 

# %% 

# General functions to interface with Matlab

# Functions with specific details on this matlab interface in particular

""" 
Here is an high-level description of each function and how to use them.
Disclamer:
   Unfortunately, the implementation overhead, given my programming experience,
was too big to isolate each functionality perfectly in such way that this
interface would be able to be 100% separable from the specifics of the Matlab
instance it was interacting with. As such, most functions have a specific part,
but the difficult and general interfacing principles are evident and can be
easily reused.

Main Interfacing functions:
    - start_matlab_instance
    - run_matlab
    - monitor_logs / any_instance_busy

Auxiliar:
    - get_instance_execution_path
    - get_log_file_path
    - get_instance_info
    - backup_log_files
    - clear_log_files
    
General Utils:
    - read_mat
    - read_compressed_mat
    
    
Specific Wrappers:
    - run_matlab_setup
    - run_matlab_save_setup
    - run_matlab_channel_calculation

- The Main interfacing functions do the following:
    i) start_matlab_instance: calls a
subprocess in the terminal with certain arguments. Is important to note that 
the directory where the set the subprocess to, will be the instant execution
directory, which is where the log file will go. So, we want to call the instant
from the log_files directory;
    ii) run_matlab calls the subprocesses in a parallel way, creating as many
of them as we set.
    iii) monitor_logs is our way to interface with matlab mid execution. Matlab
writes what's printed to the console to the log files and that way we know 
when there's an error, or when the instance is finished (by reading a 'Done'
message in the last line); any_instance_busy does something similar: returns
True if an instance is still busy, which is useful to know when to run more 
batches;

- The Auxiliar functions: 
    i) get_instance_execution_path returns the directory where the log_file 
should be left at, which is where the instance is going to be called from;
    ii) get_log_file_path is the actual specific log_file path path, uses the 
function in i);
    iii) get_instance_info is very specific: it returns a string that will be
one of the input arguments for the matlab instance. To understand this function
one needs to understand what matlab expects when. Reading the instance_info
explanation at the end of this file may help considerably;
    iv) backup_log_files copies the logs from the initial directory where they
are created, to a backup directory made only for these backups.
    v) clear_log_files clears all log files from the instances execution paths.
Since the instance monitoring is done through the log files, at the beginning 
of the next run, if the log files are still there, the monitoring process will
think those are the log files from the current instances and see the 'Done' 
and continue the execution, instead of waiting for the proper instances to
finish.

- General utils, the first is used to read a normal .mat file, the second for
mat files that have been compressed with version 7.3 or above. Therefore, the
first should be used for normal files, the second for channel coefficients, 
since this sort of file will be much larger.

- The wrappers, call the run_matlab function with specific arguments. Each
of these wrappers provides a different argument to the matlab execution, which
enables different parts of the code. They serve only to simplify and to make
the code more readable from the API perspective.


A final AND IMPORTANT note: another big part of the interface is the passage
of parameters to matlab, through other ways besides arguments. An input file
with the parameters that matlab should use to run the simulation is saved
with the function set_and_save_matlab_parameters from the classes_and_utils 
module.
"""


"""
Worth noting that the log path is where the executable is going to be called
from, because the instance log file is always placed in the directory where the
instance is called from
"""


def get_instance_execution_path(log_dir, identifier):

    return ''.join([log_dir, 'Instance', str(identifier)])


def get_log_file_path(identifier, log_dir, log_name):
    
    return ''.join([get_instance_execution_path(log_dir, identifier),
                    '\\', log_name])


def get_instance_info(flow_control, i, n_time_div, parallelisation_level):
    """ 
    Needs to return a string like '[0,30]', where the first number is the
    numeric parallelisation_level()
    """
    
    if parallelisation_level == 'None':
        paral_lvl = 0
    elif parallelisation_level == 'FR':
        paral_lvl = 1
    elif parallelisation_level == 'BS':
        paral_lvl = 2
    elif parallelisation_level == 'UE':
        paral_lvl = 3
    else:
        # parallelisation_level is only not defined when it's not going to be
        # needed. Hence, paral_lvl just needs to be defined, doesn't matter
        # it's value since it will be ignored. 
        paral_lvl = 0
    
    if flow_control == 1 or flow_control == 2:  # 'setup' or 'save_setup'
        inst_arg_2 = n_time_div
    elif flow_control == 3:
        inst_arg_2 = i
    elif flow_control == 4:
        # parallelisation_level = '-1' means blocking in series
        if parallelisation_level == '-1':
            # when flow_control = 4, instance_info(2) = 0 means 
            # 'compute for all builders'
            inst_arg_2 = 0 
        else:
            inst_arg_2 = i
    else:
        # Doesn't matter
        inst_arg_2 = 9
    # the batch_id variable is only used when doing the final step of channel
    # conversion
    
    s = ''.join(['[', str(paral_lvl), ',', str(inst_arg_2), ']'])
    
    return s
    

def start_matlab_instance(exe_path, input_filename, flow_control, instance_info, 
                          execution_path, window_on):
    # Starts a process with a matlab instance

    if isinstance(flow_control, int):
        flow_control = str(flow_control)
        
    # Formats the input arguments to be received from matlab    
    arg_str = ' '.join([input_filename,
                        flow_control, 
                        instance_info])

    # For the moment, let's keep the 'console' mode the only active.
    # To change it, we need to pass the monitoring method down the chain.
    
    command_to_call_instance = exe_path + ' ' + arg_str
    
    if window_on:
        subprocess.Popen(command_to_call_instance, 
                         creationflags=subprocess.CREATE_NEW_CONSOLE,
                         cwd=execution_path)
    else:
        subprocess.Popen(command_to_call_instance, cwd=execution_path)
    

def backup_log_files(inst_list, log_dir, log_name, dest_dir, extra_tag=''):
    
    log_files = [get_log_file_path(i, log_dir, log_name)
                 for i in inst_list]
    
    attempts_of_new_dir = 0
    original_dir_name = dest_dir
    while os.path.exists(dest_dir):
        attempts_of_new_dir += 1
        if dest_dir == original_dir_name:
            dest_dir = dest_dir + f"_{attempts_of_new_dir:2}"
        else:
            dest_dir = dest_dir[len(dest_dir)-2:] + f"{attempts_of_new_dir:2}"
    
    os.makedirs(dest_dir)
        
    for log_file in log_files:
        if os.path.isfile(log_file):
            instance_tag = '_' + log_file.split('\\')[-2] + '_'
            
            new_file_path = (dest_dir + log_name.split('.')[0] + 
                             instance_tag + log_name.split('.')[1])
                             
            if os.path.isfile(new_file_path):
                print('Warning!!!!! There already is a file with this name '
                      'in the backup directory! IT WILL NOT be replaced.')
            else:
                shutil.copy(log_file, new_file_path)
    

def clear_log_files(inst_list, log_dir, log_name):
    """ Clear past log files that still are in the directory. """
    
    # Important: before running the next batch, clean the logs of the previous
    log_files = [get_log_file_path(i, log_dir, log_name) for i in inst_list]
    
    for log_file in log_files:
        # print(log_file)
        if os.path.isfile(log_file):
            file_removed = False
            while not file_removed:
                try:
                    os.remove(log_file)
                    file_removed = True
                except PermissionError:
                    # some file handles may not been released by instances 
                    # that are still closing, hence insisting...
                    time.sleep(0.5)
                    continue
                
           
def get_instances_in_batch(n_inst_running_parallel, first_inst, last_inst,
                           n_instances, batch_id):
    """
    Returns the IDs of the instances in the batch.
    """
    instance_offset = first_inst
    
    
    if batch_id * n_inst_running_parallel > n_instances:
        n_instances_curr_batch = (n_instances - 
                                  (batch_id - 1) * n_inst_running_parallel)
    else:
        n_instances_curr_batch = n_inst_running_parallel
    
    
    offset = (batch_id - 1) * n_inst_running_parallel + instance_offset
    
    list_instances_to_run = [i for i in 
                             range(offset,
                                   offset + n_instances_curr_batch)
                             if i <= last_inst ]

    return list_instances_to_run


def any_instance_busy(inst_list, log_dir, log_name):
    # Randomly selects instances in order to check if there's any still busy
    
    instances_running = [get_log_file_path(i, log_dir, log_name) 
                         for i in inst_list]
    found_one_busy = False
    
    while not found_one_busy and instances_running != []:
        inst = random.choice(instances_running)
        # print(f"Testing instance: {inst[-17:-8]}", end=' ')
        
        try:
            with open(inst) as inst_file:
                lines = inst_file.readlines()
                if len(lines) == 0:
                    # file is empty (so, definitely busy)
                    found_one_busy = True
                else:
                    # file not empty
                    if lines[-1] == 'Done.\n':
                        instances_running.remove(inst)
                    elif "Error in Meeting" in str(lines):
                        raise Exception(lines)  
                    elif lines[-1] == 'Builder not found!\n':
                        print(f"Instance {inst[-17:-8]} "
                              f"has no builder!")
                        instances_running.remove(inst)
                        raise Exception(lines)
                    elif "Out of memory" in str(lines):
                        raise Exception('OUT OF MEMORY in instance: ' + inst)
                    else:
                        found_one_busy = True
                    
            
        except FileNotFoundError:
            # Indeed is possible to attempt to access the file before it has 
            # been created..
            # The other scenario where the file may be missing is if it has
            # been deleted in the middle of the simulation. Unless done 
            # manually, this is impossible because the only deletion calls 
            # happen right before the execution of a new batch.
            # Therefore, this covers the first case only and if you want to 
            # set the simulator in a loop, just delete one of the files mid-sim
            # (this can be fixed by putting a file back, with a line at the
            # end 'Done\n' or 'Builder not found!\n') - and you better do it
            # before the other instances finish, otherwise, just restart the 
            # simulation in the first instance of the batch that went wrong
            # print('File Not Found')
            found_one_busy = True
    
    return found_one_busy


def any_instance_done(inst_list, log_dir, log_name):
    """
    Returns the first instance it finds (from the instance list) that has 
    finished its load.
    """
    
    instances_running = [get_log_file_path(i, log_dir, log_name) 
                         for i in inst_list]
    
    if instances_running == []:
        print('No instances running!')
        return -2
        
    found_one_done = False
    inst_idx = 0
    for inst_idx in range(len(instances_running)):
        inst = instances_running[inst_idx]
        # print(f"Testing instance: {inst[-17:-8]}", end=' ')
        
        # If a busy instance is found, check the next one
        try:
            with open(inst) as inst_file:
                lines = inst_file.readlines()
                if len(lines) == 0:
                    # file is empty (so, definitely busy)
                    continue
                else:
                    # file not empty
                    if lines[-1] == 'Done.\n':
                        # Found one done!!
                        found_one_done = True
                        break
                    elif "Error" in str(lines):
                        # To add robustness!
                        raise Exception(lines)
                    elif "Error in Meeting" in str(lines):
                        raise Exception(lines)  
                    elif lines[-1] == 'Builder not found!\n':
                        print(f"Instance {inst[-17:-8]} "
                              f"has no builder!")
                        instances_running.remove(inst)
                        raise Exception(lines)
                    elif "Out of memory" in str(lines):
                        raise Exception('OUT OF MEMORY in instance: ' + inst)
                    else:
                        # If something else is written on the file, it's 
                        # probably running
                        continue
        except FileNotFoundError:
            # This instance is busy creating the file, check the next one
            continue
    
    if found_one_done:
        return inst_idx
    else:
        # print('All still running...', end='\r')
        return -1


def run_matlab(executable_path, inst_path, flow_control, input_path, 
               list_of_instances_to_run, n_time_div,
               parallelisation_level, window_on_instances):
    """ Runs a set of matlab instances in a parallel manner """
    
    # If there's no dependence with the instance ID, in the arguments of 
    # that instance, only the number of elements of the
    # list_of_instances_to_run matters
    
    # print(
    # f"{executable_path}\n {input_path} \n {flow_control} \n"
    # f"{get_instance_info(flow_control, 2, n_time_div, 
    #                      parallelisation_level)} \n"
    # f"{get_instance_execution_path(inst_path, 2)} \n"
    # f"{window_on_instances[list_of_instances_to_run.index(2)]}")
    
    
    # This can't be there, don't know why:n_jobs=len(list_of_instances_to_rund)
    
    # n_jobs is the number of parallel jobs. if <0, then (n_cpus + 1 + n_jobs).
    Parallel()(
        delayed(start_matlab_instance)(executable_path,
                                       input_path,
                                       flow_control, 
                                       get_instance_info(
                                           flow_control, i, n_time_div,
                                           parallelisation_level),
                                       get_instance_execution_path(
                                           inst_path, i),
                                       window_on_instances[
                                           list_of_instances_to_run.index(i)])
        for i in list_of_instances_to_run)
    

# ##### Some Wrappers:

def run_matlab_setup(executable_path, instance_path, input_path):
    # Runs matlab setup
    run_matlab(executable_path, instance_path, 1,  # 'setup' is 1
               input_path, 'None', [1], 0, 0, [1])
    

def run_matlab_save_setup(executable_path, instance_path, input_path, 
                          n_time_div, parallelisation_level):

    # Runs matlab setup and saves variables+builder
    run_matlab(executable_path, instance_path, 2,  # 'save setup is 2 here
               input_path, [1], n_time_div, 
               parallelisation_level, [1])


def run_matlab_channel_calculation(executable_path, instance_path, vars_path, 
                                   inst_to_run, parallelisation_level, 
                                   window_mode='windowless'):
    
    #window_mode == 'windowless':
    list_instances_window_on = [0 for i in range(len(inst_to_run) + 1)]
    
    if window_mode == 'all':
        list_instances_window_on = [1] * len(list_instances_window_on)
    if window_mode == 'first':
        list_instances_window_on[0] = 1
        
    # Runs matlab channel generation step, using the saved variables+builder
                                 # 'calculate_channel' is 3
    run_matlab(executable_path, instance_path, 3, 
               vars_path, inst_to_run, 0, 
               parallelisation_level, list_instances_window_on)
    

def run_matlab_blocking_computation_series(executable_path, instance_path, 
                                           vars_path):

    run_matlab(executable_path, instance_path, 4, 
               vars_path, [1], 0, '-1', [1])


def run_matlab_blocking_computation_parallel(executable_path, instance_path, 
                                           vars_path, parallelisation_level):

    run_matlab(executable_path, instance_path, 4, 
               vars_path, [1], 0, parallelisation_level, [1])


def run_matlab_aggregate_channels(executable_path, instance_path, 
                                  vars_path):

    run_matlab(executable_path, instance_path, 5, 
               vars_path, [1], 0, 0, [1])




# This function has fell out of use. But it's left here as an utility.
def monitor_logs(instances_to_monitor, log_dir, log_name, 
                 p1=1, p2=1, print_instance=1, verbose=0):
    """ Logs Monitoring loop - only allows the program continuation after the
    last log file has been written with 'Done' at the end."""
    
    every_instance_done = 0
    done_instances = []
    
    instances = [get_log_file_path(i, log_dir, log_name)
                 for i in range(1, instances_to_monitor + 1)]
    
    
    while not every_instance_done:
        for i in range(len(instances)):
            inst = instances[i]
            # Don't print a finished instance
            if inst in done_instances:
                continue
            
            # Print and monitor unfinished instances
            try:
                with open(inst) as log_fp:
                    if verbose:
                        print(f"Instance {i+1} is still running...")
                    lines = log_fp.readlines()
                    if len(lines) > 0:
                        if lines[-1] == 'Done.\n':
                            done_instances.append(inst)
                        
                        if "Error" in str(lines):
                            raise Exception(lines)
                        
                        if len(lines) >= 2 and \
                                lines[-2] == 'Too many input arguments.\n':
                            raise Exception('No folder in path'
                                            ' can have spaces!')
                    
                        if lines[-1] == 'Builder not found!\n':
                            print(f"Instance {i} couldn't find its builder.")
                            done_instances.append(inst)
                            
                    if print_instance == 1:
                        log_fp.seek(0)
                        print(log_fp.read())
                        
            except FileNotFoundError:
                # File wasn't created yet
                if verbose:
                    print('Initialising...')
                continue
            # Pause in between instances
            time.sleep(p1)
        
        
        # Pause also in between loops
        if verbose:
            print('Also pause in between loops.')
        time.sleep(p2)
        
        if len(done_instances) == instances_to_monitor:
            every_instance_done = 1
        
        """
        Note: This function can be optimised by checking only the last line 
        instead of reading all lines and picking the last. The files should be 
        very small, so there will probably be no impact. Nonetheless, glenbot 
        has a good solution at: https://stackoverflow.com/questions/136168/
        """


def clear_channel_parts():
    """ 
    Goes to the channel file and deletes the past channels with the same
    name as the current channels that need to be computed
    """
    # this probably isn't needed since matlab overwrites them
    
    # The only thing it may help avoid is using channel parts from 2 different
    # simulations, but this can be checked while putting the files together.
    pass


# %% General utils


def read_mat(fname):
    # This function reads Matlab variables from a .mat file (versions <= 7.2)
    # Basically, everything but coefficients, those are compressed (v7.3)
    ut.parse_input_type(fname, 'str')
    
    f_len = len(fname)

    if f_len > 4 and fname[f_len - 4:f_len] == '.mat':
        return scipy.io.loadmat(fname) 
    else:
        print('File doens\'t exist or needs .mat at the end.')


def read_compressed_mat(fname, var_to_read='freqresp'):
    # This function reads Matlab variables saved in version 7.3 (hdf5 format)
    # More used to channel coefficients
    ut.parse_input_type(fname, 'str')
    ut.parse_input_type(var_to_read, 'str')
    
    f_len = len(fname)

    if f_len > 4 and fname[f_len - 4:f_len] == '.mat':
        f = h5py.File(fname, 'r')
        data = f.get(var_to_read)
        return np.array(data) 
    else:
        print('File doens\'t exist or needs .mat at the end.')


def read_matrix_binary(file):
    """
    Read a file in binary to a complex numpy matrix.
    """
    
    real_part = np.fromfile(file + '_r.bin', dtype=np.float64)
    imag_part = np.fromfile(file + '_i.bin', dtype=np.float64)
    
    
    return (np.ones(3, dtype=np.complex64) * real_part + 
            np.ones(3, dtype=np.complex64) * imag_part * 1j)


# %% Explanation of some variables that interface with matlab

# Variable: flow_control 
# This should be passed as argument to the MATLAB instance, so that
# only the important part of the script runs (one may need to adjust the 
# configurations carefully before starting the channel calculation)
# Options:
# flow_control = 0 (i.e 'complete' mode: runs everything)
# flow_control = 1 (i.e 'setup' mode: executes only the setup, useful for
#                       fine-tuning the simulation parameters using plots
# flow_control = 2 (i.e 'save_setup' mode: same as 1, but also
#                       saves all the variables.
# flow_control = 3 (i.e 'calculate_channel' mode (to be used after 2))
#                       Computes channel in a parallel state
# the input and output paths have different roles depending on the execution
# mode (control variable):
# 0 - input parameters, output complete channel.
# 1 - input parameters
# 2 - input parameters, variables save file
# 3 - variables save file, output channel part preffix, e.g. 'ch'.

# Variable: instance_info
# The first position is used for the parallelisation level [0-3]
# this parameter is used along with parameters 2 to determine the
# amount of builders to create and which builder to load and use in the
# channel calculation phase
# The second parameter has 2 functions, depending    
# 1- in the setup/save_setup fase to tell matlab in how many chunks
# it needs to divide each user's track in order to improve parallel
# performance. From the number of instances and the first parameter,
# is possible to derive the amount of division in time.
# 2- in the channel calculation phase, to tell matlab the high level
# instance to use for the computation. Here, the first parameters comes
# into play to derive which file to load and which builder to use.

# The first parameter is only used at the setup but the second is used both
# at the setup and at the calculation steps.




