#!/usr/bin/env python

""" 
TO RUN:

conda activate RayRLlib
cd /home/yaoyao/Documents/SimpackFiles/04_GeneralModel
python Model2SPCK_MultiClients.py                                         
 
conda activate RayRLlib
cd /home/yaoyao/Documents/SimpackFiles/04_GeneralModel
tensorboard --port 6689 --logdir  ~/ray_results/APEX_DDPG/                                                                                                                                             

"""
import subprocess
import threading
import time
import Model2SPCK_helper as helper

config_values = helper.ReadConfig()
CalLogName = config_values.CalLogName
ResetPortTimeout = config_values.ResetPortTimeout
script_path = config_values.Client_path

this_Workers = config_values.this_Workers
this_StartWith = config_values.this_StartWith


start_port = this_StartWith + 9900
end_port = start_port + this_Workers 
processes = {}
 
for port in range(start_port, end_port):
    startPorts = f"python {script_path} --port {port}"
    process = subprocess.Popen(startPorts, shell=True)
    processes[port] = {
        'process': process,
        'last_update': time.time(), 
    }

monitor_thread = threading.Thread(target=helper.monitor_processes, args=(processes, script_path, CalLogName, this_Workers, this_StartWith, ResetPortTimeout))
monitor_thread.start()

for process_info in processes.values():
    process_info['process'].wait()

