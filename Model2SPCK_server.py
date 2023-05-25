#!/usr/bin/env python

""" 
TO RUN:

conda activate RayRLlib
cd /home/yaoyao/Documents/SimpackFiles/04_GeneralModel
python Model2SPCK_server.py 


|编号	 |与策略Server的HTTP通信端口 |Simpack内部计算的UDP通信端口 |与SimpackRT的TCP通信启动端口A |与SimpackRT的TCP循环通信端口B |
|  ---- | ----                    |  ----                    | ----                      |  ----                     | 
|00	    |9900	                  |12900	                 |9600	                     |9100	                     |
|01	    |9901	                  |12901	                 |9601	                     |9101	                     |
|02	    |9902	                  |12902	                 |9602	                     |9102	                     |
|03	    |9903	                  |12903	                 |9603	                     |9103	                     |

"""
import ray
from ray import air, tune
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.examples.custom_metrics_and_callbacks import MyCallbacks
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.apex_ddpg.apex_ddpg import ApexDDPGConfig

import argparse
import gymnasium as gym
import numpy as np
import threading
import time
from filelock import FileLock
# 导入自定义函数
import Model2SPCK_helper as helper

DEFULT_SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9900  # + worker-idx - 1
CHECKPOINT_FILE = "last_checkpoint_{}.out"

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--port-base",
        type=int,
        default=SERVER_BASE_PORT,
        help="The base-port to use (on localhost). " f"Default is {SERVER_BASE_PORT}.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=5_000_000, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=5_000_000,
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=5_000_000,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_cli_args()
    config_values = helper.ReadConfig() 
    isRandomInit = config_values.isRandomInit
    # Server_workers表示Server启动的端口数
    Server_workers =  config_values.Server_workers
    # this_Workers表示本机启动的Client端口数，其他计算机上也会启动仿真端口
    this_Workers = config_values.this_Workers
    # this_StartWith表示本机启动的初始端口相对于9900的偏移地址
    this_StartWith = config_values.this_StartWith
    # mainServer_IP4表示主机地址的末尾，server的IP地址为192.168.1.mainServer_IP4
    mainServer_IP4 = config_values.mainServer_IP4
    SERVER_ADDRESS = f"192.168.1.{mainServer_IP4}"
        
    CalLogName = config_values.CalLogName
    # 删除原有SPCK与相关文件,但保留 Comm2SPCK.c' 与 'Comm2SPCK
    folder_path = 'ParallelSPCKs'  # 文件夹路径
    excluded_files = ['Comm2SPCK.c', 'Comm2SPCK']  # 排除的文件
    helper.CleanParallelFolder(folder_path,excluded_files);
    time.sleep(0.5)
    
    # 分发SPCK与subvars文件
    helper.copy_and_modify_files('Model.spck', folder_path) # SPCK文件
    helper.copy_files('Model_Subvars.subvar', folder_path)   # subvars文件
    time.sleep(0.5)
    
    # 新建文本文件,以记录每次SPCK client的时间
    with open(CalLogName, 'w') as f:
        for _ in range(50):
            f.write("\n")    
            
    # # 创建并开始另一个线程上以一定时间间隔重置subvars_X
    if (isRandomInit == 1):
        SimpackRandomInit = threading.Thread(target=helper.SubvarsAlwaysRandomInit, args=(this_StartWith,this_Workers,2)) # IntervalTIme = 2
        SimpackRandomInit.start()
        
    # 串行启动SPCK
    # 启动SPCK应在调试完成后放置于Model2SPCK_MultiClients.py内
    print("\n\n启动Server所在主机的SPCK\n\n")
    for port in range(args.port_base + this_StartWith , args.port_base + this_StartWith + this_Workers):
        print("\n本机起始端口号:",args.port_base + this_StartWith)
        print("START No.", port - args.port_base  + 1,"PORT among all Clients Computers")
        
        if (helper.checkTCPA(port)):
            print(f"TCP port {helper.AllCOMMPORTS(port).TCPportA_2_SPCK} is in use, IGNORE STARTING THIS PORT.")
        else:
            helper.OPEN_TCPA_SPCKrt(port) 
        time.sleep(0.5)
    
    # 启动Ray
    ray.init()
    
    def _input(ioctx):
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                SERVER_ADDRESS,
                args.port_base + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        else:
            return None
    
    OB_low  = np.array([-1,-100,-2.5,-20]).astype(np.float32)
    OB_high = np.array([+1,+100,+2.5,+20]).astype(np.float32)
    At_low  = np.array([-1,]).astype(np.float32)
    At_high = np.array([+1,]).astype(np.float32)
    
    config = (
        ApexDDPGConfig()
        .environment(
            env = None,
            observation_space = gym.spaces.Box(low=OB_low, high=OB_high, shape=(4,)),
            action_space = gym.spaces.Box(low=At_low, high=At_high, shape=(1,)),
        )
        .framework("torch")
        .offline_data(input_=_input)
        .rollouts(
            num_rollout_workers = Server_workers,
            enable_connectors = False,
        )
        .resources(num_gpus = 1)  
        .evaluation(off_policy_estimation_methods = {})
        .debugging(log_level = "INFO")
        .training(tau = 0.5, use_huber=True, n_step = 1, target_network_update_freq = 2048) 
    )
    config.rl_module(_enable_rl_module_api=False)
    config.training(_enable_learner_api=False)
    config.replay_buffer_config["capacity"] = 2_500_000 
    config["rollout_fragment_length"] = 32  
    config["smooth_target_policy"] = True
    config["train_batch_size"] = 512  
    config["num_learner_workers"] = 5
    config["num_steps_sampled_before_learning_starts"] = 2048 
    config["clip_actions"] = True
    
# config["clip_rewards"] = False
# config["observation_filter"] = "MeanStdFilter"    
#.resources(num_trainer_workers=5)

    # 记录神经网络训练配置
    config_dict = config.to_dict()
    with open("ParallelSPCKs/TrainingConfigs.txt", "a") as file:
        print(config_dict, file=file)

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    
    # checkpoint_config = air.CheckpointConfig(checkpoint_frequency=5,checkpoint_at_end=True)
    
    tune.Tuner(
        "APEX_DDPG", 
        param_space=config, 
        run_config=air.RunConfig(stop=stop, \
            verbose=2, \
            # verbose = 2 可以看到isalive错误的真实原因————超过观测空间
            #checkpoint_config=checkpoint_config, \
            #local_dir="/home/yaoyao/Documents/RayRL/04_myRLenv/Checkpoints"
        )
    ).fit()