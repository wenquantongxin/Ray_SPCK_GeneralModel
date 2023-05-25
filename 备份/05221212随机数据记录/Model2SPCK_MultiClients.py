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
num_workers = config_values.num_workers
CalLogName = config_values.CalLogName
ResetPortTimeout = config_values.ResetPortTimeout
script_path = config_values.Client_path

start_port = 9900
end_port = start_port + num_workers 
processes = {}
 
for port in range(start_port, end_port):
    startPorts = f"python {script_path} --port {port}"
    process = subprocess.Popen(startPorts, shell=True)
    processes[port] = {
        'process': process,
        'last_update': time.time(), 
    }

monitor_thread = threading.Thread(target=helper.monitor_processes, args=(processes, script_path, CalLogName, ResetPortTimeout))
monitor_thread.start()

for process_info in processes.values():
    process_info['process'].wait()

