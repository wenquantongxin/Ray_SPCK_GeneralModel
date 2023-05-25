#!/usr/bin/env python

""" 
TO RUN:

conda activate RayRLlib
cd /home/yaoyao/Documents/SimpackFiles/04_GeneralModel
python Model2SPCK_client.py  --port 9900                                              

 
conda activate RayRLlib
cd /home/yaoyao/Documents/SimpackFiles/04_GeneralModel
tensorboard --port 6689 --logdir  ~/ray_results/APEX_DDPG/                 

"""

import argparse
from ray.rllib.env.policy_client import PolicyClient
import numpy as np
import pandas as pd
import Model2SPCK_helper as helper
from Model2SPCK_helper import SPCKenv


parser = argparse.ArgumentParser()
parser.add_argument(
    "--port", type=int, default=9900, help="The port to use (on localhost) connecting to Policy Server."
)

if __name__ == "__main__":
    args = parser.parse_args()
    SPCKCP_Obj = SPCKenv(args.port)
    
    config_values = helper.ReadConfig() 
    mainServer_IP4 = config_values.mainServer_IP4
    SERVER_ADDRESS = f"192.168.1.{mainServer_IP4}"

    client = PolicyClient(
        f"http://{SERVER_ADDRESS}:{args.port}", inference_mode = "local"
    )
    
    # Start a new episode.
    obs, info = SPCKCP_Obj.reset()
    eid = client.start_episode()
    rewards = 0.0
    while True:
        action = client.get_action(eid, obs)

        # Perform a step in the external simulator (env).
        obs, reward, terminated, truncated, info = SPCKCP_Obj.step(action)
        rewards += reward

        # Log next-obs, rewards, and infos.
        client.log_returns(eid, reward, info=info)

        # Reset the episode if done.
        if terminated or truncated:
          
            print("Python: 回合总Reward:", rewards,"\n")
            rewards = 0.0
            
            # End the old episode.
            client.end_episode(eid, obs)

            # Start a new episode.
            obs, info = SPCKCP_Obj.reset()
            
            eid = client.start_episode()
            
            
