# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:42:34 2020

@author: janeiroja
"""
# %% Imports

# Imports of standard Python Libraries
import time
import numpy as np


# Own code imports
import utils as ut
import matlab_interface as mi
import generation_parameters as gen_par


usage = """sxr_gen.py <instances in parallel> [first instance]
        [last instance] <first seed> <last seed> <speed>\n
        Optional arguments are within [] and depend on specify_instances
        and read_inputs_from_command_line generation_variables. 
        The others (within <>) are mandatory.
        
        PS: for now, input first and last instance...."""
                   
try:
    first_seed = int(ut.get_input_arg(-3))
    last_seed = int(ut.get_input_arg(-2))
    speed = int(ut.get_input_arg(-1))
    if not (first_seed <= last_seed and first_seed > 0 and last_seed > 0 \
            and speed >= 0):
        print(f'first_seed = {first_seed}, of type {type(first_seed)}')
        print(f'last_seed = {last_seed}, of type {type(last_seed)}')
        print(f'speed = {speed}, of type {type(speed)}')
        print(f'Usage: {usage}')
        ut.stop_execution()
except Exception as e:
    print(e)
    print(f'Usage: {usage}')
    ut.stop_execution()
    
seeds = [i for i in range(first_seed, last_seed + 1)]

print(f'Generating for SEEDS: {seeds}, SPEED {speed}')

#for s in seeds: 
for seed in seeds:
    
    run_id_str = f'SEED-{seed}_SPEED-{speed}'
    
    print(f'Starting generation for {run_id_str}.')    
    
    # Initialise the generation parameters
    sp = gen_par.Generation_parameters(seed, speed)
    print('Done setup of generation parameters.')
    
    # Start timer per seed generation
    t_0 = time.time()
    
    # ####### Save Setup and variables and make Channel Builders ###########
    
    # Run the save_setup procedure
    if not sp.use_existing_builders:
        print('Begin generating builders...')
    
        t = time.time()
        mi.clear_log_files([1], sp.log_dir, sp.log_file_name)
        mi.run_matlab_save_setup(sp.executable_path, 
                                 sp.log_dir,
                                 sp.input_param_file,
                                 sp.time_divisions, 
                                 sp.parallelisation_level)
        
        mi.monitor_logs(1, sp.log_dir, sp.log_file_name, 
                        p2 = 0.1, print_instance=0)
        
        if sp.backup_logfiles:
            # Backup the log of the setup - for first instance only.
            mi.backup_log_files([1], 
                                sp.log_dir, 
                                sp.log_file_name,
                                sp.log_backup_dir, 'Setup')
        
        t_setup = round(time.time()-t,1)
        print(f"Setup finished. Time enlapsed: {t_setup} s")
    else:
        t_setup = 0
        
    if sp.only_setup:
        ut.stop_execution()
    
    
    # ########## Run batch(es) of instances in parallel #############
    
    if sp.dry_run:
        print('Attention! This is a dry-run, nothing will be computed!')
    
    # See in parameter setting the difference between hard and soft batching.
    # In short, soft is more efficient and flexible in terms of parallelisation
    # settings, but hard is more organised, debugs easier and has progress bar
    if sp.batching_strat == 'hard_batching':
        # To monitor the batch progress, based how long the last batch took
        last_batch_duration = 0  # [s]
        
        
        for batch in range(1, sp.n_batches + 1):
            
            print(f"Running batch {batch}")
            
            inst_in_batch = \
                mi.get_instances_in_batch(sp.n_inst_running_parallel,
                                          sp.first_instance,
                                          sp.last_instance,
                                          sp.n_total_instances, batch)
            
            print(f"Starting instances: {inst_in_batch}")        
            
            if sp.dry_run:
                continue
            
            # Delete existent log files
            mi.clear_log_files(inst_in_batch,
                               sp.log_dir, 
                               sp.log_file_name)
            
            t_start_batch = time.time()
            
            # Run instances in parallel
            mi.run_matlab_channel_calculation(sp.executable_path, 
                                              sp.log_dir,
                                              sp.gen_folder,
                                              inst_in_batch, 
                                              sp.parallelisation_level,
                                              sp.window_mode)
            
            while mi.any_instance_busy(inst_in_batch, 
                                       sp.log_dir,
                                       sp.log_file_name):
                
                # Print progress if that data is available
                if last_batch_duration != 0:
                    time_enlapsed = time.time() - t_start_batch
                    print("Progress: " +
                          "{:.2f}".format(time_enlapsed / 
                                          last_batch_duration * 100) + '%', 
                          end='\r')
                
                time.sleep(3)
                
                
            
            last_batch_duration = time.time() - t_start_batch
            print(f"Batch {batch} Done! "
                  f"Time Taken: {round(last_batch_duration,2)}s.")
        
            
            if sp.backup_logfiles:
                mi.backup_log_files(inst_in_batch, 
                                    sp.log_dir, 
                                    sp.log_file_name,
                                    sp.log_backup_dir, 'Channel')    
                print('Backup of logfiles complete.')
                
    elif sp.batching_strat == 'soft_batching' and not sp.dry_run:
        
        inst_running = []
        last_instance_started = -1
        
        # Delete existent log files of that instance
        mi.clear_log_files(np.arange(sp.first_instance, sp.last_instance + 1),
                           sp.log_dir, 
                           sp.log_file_name)
        
        while last_instance_started < sp.last_instance:
            
            # Don't rush instance creation. Sometimes they take a couple of
            # seconds to release the memory, no need to create them that fast.
            time.sleep(1)
            
            # Check if any instance has finished in the meantime
            inst_done_idx = mi.any_instance_done(inst_running, sp.log_dir,
                                                 sp.log_file_name)
            
            if inst_done_idx == -1:
                # All instances are running...
                # Give some time to some instance to finish
                if len(inst_running) == sp.n_inst_running_parallel:
                    time.sleep(3)
                    continue
                # else, create a new instance!
            elif inst_done_idx >= 0:
                # One has finished!
                print(f'Finished instance: {inst_running[inst_done_idx]}')
                inst_running.remove(inst_running[inst_done_idx])
                
                # Backup the log of the finished instance 
                if sp.backup_logfiles:
                    mi.backup_log_files(inst_running[inst_done_idx], 
                                        sp.log_dir, 
                                        sp.log_file_name,
                                        sp.log_backup_dir, 'Channel')    
                    print(f'Backup of instance {inst_running[inst_done_idx]} '
                          f'logfile complete.')
            
            
            if (len(inst_running) < sp.n_inst_running_parallel and
                last_instance_started < sp.last_instance):
                # Start a new one:
                    
                if inst_running == []:
                    instance_to_start = sp.first_instance
                else:
                    instance_to_start = inst_running[-1] + 1
                
                if instance_to_start > sp.last_instance:
                    # You've gone too far, don't start this one!
                    continue
                
                print(f'Starting instance: {instance_to_start}')
                inst_running.append(instance_to_start)
                
                
                # To be extra sure the file is clean, to avoid extra loops
                mi.clear_log_files([instance_to_start],
                                   sp.log_dir, 
                                   sp.log_file_name)
                
                
                # Run instance in parallel
                mi.run_matlab_channel_calculation(sp.executable_path, 
                                                  sp.log_dir,
                                                  sp.gen_folder,
                                                  [instance_to_start], 
                                                  sp.parallelisation_level,
                                                  sp.window_mode)
                
                last_instance_started = instance_to_start
            
                
        print('The last instance has been started! '
              'Waiting for remaining instances to finish...')
        # Be sure the last instances are done:
        while inst_running != []:
            # Check if any instance has finished in the meantime
            inst_done_idx = mi.any_instance_done(inst_running, sp.log_dir,
                                                 sp.log_file_name)
            if inst_done_idx >= 0:
                print(f'Finished instance: {inst_running[inst_done_idx]}.')
                inst_running.remove(inst_running[inst_done_idx])
            
            if inst_running != []:
                # Wait a bit before checking again
                time.sleep(0.5)
    
    print('Done Complete Channel Computation.')
    
    ########################################################################
    
    if sp.apply_blocking:
        # Compute Blocked Channel
        print('Computing Human Blockage.')
        
        mi.run_matlab_blocking_computation_series(sp.executable_path, 
                                                  sp.log_dir,
                                                  sp.gen_folder)
    
        mi.monitor_logs(1, sp.log_dir, sp.log_file_name, 
                        p2 = 0.1, print_instance=1)
        
    if sp.aggregate_channels:
        # Aggregate Channels in Time and/or Frequency Domain
        print('Aggregating channels.')
        
        mi.run_matlab_aggregate_channels(sp.executable_path, 
                                         sp.log_dir,
                                         sp.gen_folder)
    
        mi.monitor_logs(1, sp.log_dir, sp.log_file_name, 
                        p2 = 0.1, print_instance=1)
    
    print(f'DONE Generation of {run_id_str}.')
    
    t_total = round(time.time() - t_0)
    t_comp = round(t_total - t_setup)
    print(f'Time taken for instance computation: {t_comp} sec.')
    print(f"Total time taken for {run_id_str}: {t_total} sec.")


    if sp.delete_builders_at_end: #& sp.last_instance == sp.n_total_instances:
        # Delete builders' folder
        ut.del_dir(sp.builders_folder)
