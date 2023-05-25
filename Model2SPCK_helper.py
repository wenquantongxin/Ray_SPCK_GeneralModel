import numpy as np
import socket
import struct
import subprocess
import threading
import time
import random
import re
import pandas as pd
import shutil
import os
import signal
import math
import datetime

# Terminated为自然结束，例如达到边界条件
def isTerminated(sintheta,Xcar):
    # 若Pole的角度大于0.73rad 或 Car的位移大于2.5m,则环境提前Terminated
    # if (abs(theta) > 0.73) or (abs(Xcar) > 2.5): 
    if (abs(sintheta) >= math.sin(0.73)) or (abs(Xcar) >= 2.0):
        return 1
    else:
        return 0
    
# 判断是否Truncated
# Truncated为人为结束，例如达到最长时间
def isTruncated(TsNow,MaxTs):
    if (TsNow > MaxTs): 
        return 1
    else:
        return 0

# 计算奖励值              
def CalReward_Alive(Terminated):
    if (Terminated == False):
        reward = +1
    else:
        reward = -10
    return reward

def CalReward_State(sintheta,dottheta,Xcar):
    reward = - ( sintheta ** 2 + 0.1 * dottheta ** 2 + 0.1 *  Xcar ** 2   )
    return reward

# 根据base port计算UDP/TCP通信所需各个端口号
class COMMPORTS:
    def __init__(self):
        self.HTTPport_2_PolicyServer = None
        self.UDPport_in_SPCK = None
        self.TCPportA_2_SPCK = None
        self.TCPportB_2_SPCK = None
        self.NoPort = None
        
def AllCOMMPORTS(port_base):
    CommPorts = COMMPORTS()
    CommPorts.HTTPport_2_PolicyServer = port_base
    CommPorts.UDPport_in_SPCK = port_base + (12900-9900)
    CommPorts.TCPportA_2_SPCK = port_base + ( 9600-9900)
    CommPorts.TCPportB_2_SPCK = port_base + ( 9100-9900)
    CommPorts.NoPort = port_base - 9900 + 1
    return CommPorts

# 检查TCP-A端口是否被占用
def checkTCPA(port_base):
    TCPportA_2_SPCK = AllCOMMPORTS(port_base).TCPportA_2_SPCK 
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcpA_in_use = False
    try:
        tcp_socket.bind(("127.0.0.1", TCPportA_2_SPCK))
    except socket.error:
        tcpA_in_use = True
    tcp_socket.close()   
    return tcpA_in_use 

# 初始化
def TCP2SIMPACK_Init(TCPaddr_2_SPCK):
    buffer_size = 512  
    SERVER_ADDRESS = '127.0.0.1'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    client_socket.connect((SERVER_ADDRESS, TCPaddr_2_SPCK))
    recorded_y_values = []
    for ts in range(0, 5):
        data = client_socket.recv(4)
        ny = struct.unpack('!I', data)[0]
        y_values = []
        for _ in range(ny):
            data = client_socket.recv(4)
            y_value = struct.unpack('!f', data)[0]
            y_values.append(y_value)
        recorded_y_values.append(y_values)
        # 提前终止
        if ( ts  >  1 ):
            client_socket.close()
            break
        u_values = [0,0,0,0] 
        u_values_str = " ".join([f"{u_value:.4f}" for u_value in u_values])
        client_socket.sendall(f"{u_values_str}\n".encode('utf-8'))
    client_socket.close()

# 启动另一个线程执行cmd命令
def run_command(command):
    subprocess.run(command, shell=True)

# 使用TCP-A启动SPCK RT    
def OPEN_TCPA_SPCKrt(port_base):
    CommPorts = AllCOMMPORTS(port_base)
    UDPport_in_SPCK = CommPorts.UDPport_in_SPCK
    TCPportA_2_SPCK = CommPorts.TCPportA_2_SPCK
    config_values = ReadConfig()
    command = f"! {config_values.Work_path}/ParallelSPCKs/Comm2SPCK {TCPportA_2_SPCK} {UDPport_in_SPCK} {0.01} {1} {config_values.Work_path} {config_values.SPCK_path}"
    SPCK_pendulum_process = threading.Thread(target=run_command, args=(command,))
    SPCK_pendulum_process.start()
    time.sleep(1)
    tcp2simpack_process = threading.Thread(target=TCP2SIMPACK_Init, args=(TCPportA_2_SPCK,))
    tcp2simpack_process.start()
    tcp2simpack_process.join()    
    SPCK_pendulum_process.join()

# 读取config文件配置
class SimuConfig:
    def __init__(self):
        self.Maxtimesteps = None
        self.ControlInterval = None
        self.isRandomInit = None
        self.Server_workers = None
        self.ResetPortTimeout = None
        # self.dim_u = None
        # self.dim_y = None
        self.CalLogName = None
        self.Client_path = None
        self.SPCK_path = None
        self.Work_path = None
        self.Spck_name = None
        self.mainServer_IP4 = None
        self.this_StartWith = None
        self.this_Workers = None
        
def ReadConfig():
    config_values = SimuConfig()
    with open("Model_config.config", "r") as file:
        for line in file:
            match = re.match(r'(\w+):\s+(.*)', line.strip())
            if match:
                key, value = match.groups()
                if key in ["ControlInterval", "ResetPortTimeout"]:
                    value = float(value)
                elif key in ["Maxtimesteps", "isRandomInit", "Server_workers","mainServer_IP4","this_StartWith","this_Workers"]:
                    value = int(value)
                setattr(config_values, key, value)
    return config_values

# 写入SPCK subvars文件
def SubvarsFilesInit(this_SPCKstartid,this_SPCKNum,Reset2Zeros = False): 
    results = []
    for i in range(this_SPCKstartid + 1, this_SPCKstartid + this_SPCKNum + 1):
        if (Reset2Zeros == True): # Reset2Zeros
            var1 = 0
            var2 = 0
            var3 = 0
            var4 = 0
        else: 
            var1 = random.uniform(-0.2, 0.2)
            var2 = 0
            var3 = 0
            var4 = 0
        content = f"""!file.version=3.6! Removing this line will make the file unreadable

        !**********************************************************************
        ! SubVars
        !**********************************************************************
        subvar($_InitInput_A, str= '{var1}')                                                             ! $_InitInput_A
        subvar($_InitInput_B, str= '{var2}')                                                             ! $_InitInput_B
        subvar($_InitInput_C, str= '{var3}')                                                             ! $_InitInput_C
        subvar($_InitInput_D, str= '{var4}')                                                             ! $_InitInput_D
        """
        
        with open(f'./ParallelSPCKs/Model_Subvars_{i}.subvar', 'w') as f:
            f.write(content)   
        results.append((var1,var2,var3,var4))

# 以一定时间间隔IntervalTime重置SPCK subvars文件
def SubvarsAlwaysRandomInit(this_SPCKstartid,this_SPCKNum,IntervalTIme = 1):
    while True:
        SubvarsFilesInit(this_SPCKstartid,this_SPCKNum,Reset2Zeros = False)
        time.sleep(IntervalTIme) 
        
# 读取subvars文件初始化的值   
def ReadRandomSubvars():
    values = {}
    with open('Model_Subvars.subvar', 'r') as file:
        # 读取文件的每一行
        for line in file:
            # 使用正则表达式匹配需要的行
            match = re.match(r"subvar\((\$_InitInput_[A-D]), str= '([-0-9\.]+)'\)", line.strip())
            if match:
                values[match.group(1)] = float(match.group(2))
    return values['$_InitInput_A'], values['$_InitInput_B'], values['$_InitInput_C'], values['$_InitInput_D']

def ReadSubvars_ByPort(port_base):
    values = {}
    with open(f'./ParallelSPCKs/Model_Subvars_{AllCOMMPORTS(port_base).NoPort}.subvar', 'r') as file:
        # 读取文件的每一行
        for line in file:
            # 使用正则表达式匹配需要的行
            match = re.match(r"subvar\((\$_InitInput_[A-D]), str= '([-0-9\.]+)'\)", line.strip())
            if match:
                values[match.group(1)] = float(match.group(2))
    return values['$_InitInput_A'], values['$_InitInput_B'], values['$_InitInput_C'], values['$_InitInput_D']

# 将每个端口号上的每一个回合仿真的时间记录于Log中
def RecordLogFile(port_base,CalLogName):
    # 在每次迭代后，写入当前的port-base和时间戳
    try:
        with open(CalLogName, 'r') as f:
            lines = f.readlines()
        index = port_base - 9900  # 计算要修改的行号
        lines[index] = f"{port_base} {time.time()}\n"  # 修改对应的行
        with open(CalLogName, 'w') as f:
            f.writelines(lines)
    except IndexError:
        pass

# 检查TCP-A/B是否被占用 
def checkTCP(port_base):
    CommPorts = AllCOMMPORTS(port_base)
    TCPportA_2_SPCK = CommPorts.TCPportA_2_SPCK
    TCPportB_2_SPCK = CommPorts.TCPportB_2_SPCK
    tcpA_in_use = False
    tcpB_in_use = False
    try:
        tcp_socket_A = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket_A.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_socket_A.bind(("127.0.0.1", TCPportA_2_SPCK))
    except socket.error:
        tcpA_in_use = True
    finally:
        if not tcpA_in_use:
            tcp_socket_A.close()
    try:
        tcp_socket_B = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket_B.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_socket_B.bind(("127.0.0.1", TCPportB_2_SPCK))
    except socket.error:
        tcpB_in_use = True
    finally:
        if not tcpB_in_use:
            tcp_socket_B.close()
    return tcpA_in_use,tcpB_in_use        

# 将文件拷贝至指定路径，可以覆盖，格式为 源文件名_N        
def copy_files(filename, target_dir):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 为每个Workers创建一份复制的文件
    config_values = ReadConfig() 
    this_Workers = config_values.this_Workers
    this_StartWith = config_values.this_StartWith
    
    for i in range(this_StartWith + 1, this_StartWith + this_Workers + 1):
        # 在原始文件名和序号之间添加一个下划线
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_{i}{ext}"

        # 构造源文件路径和目标文件路径
        src = os.path.join(os.getcwd(), filename)
        dst = os.path.join(os.getcwd(), target_dir, new_filename)

        # 复制并重命名文件
        shutil.copy(src, dst)
        print(f"Copied file to: {dst}")

# 将spck文件拷贝至指定路径，修改subvars 
def copy_and_modify_files(filename, target_dir):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 为每个Workers创建一份复制的文件
    config_values = ReadConfig() 
    this_Workers = config_values.this_Workers
    this_StartWith = config_values.this_StartWith
    
    for i in range(this_StartWith + 1, this_StartWith + this_Workers + 1):
        # 在原始文件名和序号之间添加一个下划线
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_{i}{ext}"

        # 构造源文件路径和目标文件路径
        src = os.path.join(os.getcwd(), filename)
        dst = os.path.join(os.getcwd(), target_dir, new_filename)

        # 复制并重命名文件
        shutil.copy(src, dst)
        print(f"Copied file to: {dst}")

        # 打开新复制的文件并查找特定的行
        with open(dst, 'r') as file:
            lines = file.readlines()

        for j, line in enumerate(lines):
            if line.strip().startswith("subvarset.file ("):
                # 替换行
                lines[j] = line.replace('./Model_Subvars', f'./Model_Subvars_{i}')

        # 重新写入文件
        with open(dst, 'w') as file:
            file.writelines(lines)
        
# 删除子文件夹下除Comm2SPCK.c和Comm2SPCK的其他所有文件和文件夹        
# 本函数不涉及具体文件名
def CleanParallelFolder(folder_path,excluded_files):
    for filename in os.listdir(folder_path):
        if filename not in excluded_files:
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # 确保是文件而不是文件夹
                os.remove(file_path)        
            elif os.path.isdir(file_path): # 如果是文件夹
                shutil.rmtree(file_path)
        
# 通过读取端口写在Log中的时间戳,若超时则重启该端口
def monitor_processes(processes, script_path, CalLogName, this_Workers, this_StartWith, ResetPortTimeout = 100):
    while True:
        if os.path.exists(CalLogName):
            with open(CalLogName, 'r') as f:
                lines = f.readlines()
                # 根据 this_Workers 和 this_StartWith 确定检查的行数范围
                lines_to_check = lines[this_StartWith : this_StartWith + this_Workers]
                for line in lines_to_check:
                    if line.strip():  # 检查行是否为空
                        port, last_time = line.strip().split()  # 假设每行是"port timestamp"
                        port = int(port)  # 将port转换为整数
                        last_time = float(last_time)  # 将时间戳转换为浮点数
                        
                        if port in processes:
                            process_info = processes[port]
                            last_update = process_info['last_update']
                            
                            # 如果时间戳比上次更新的时间新，更新最后更新时间
                            if last_time > last_update:
                                process_info['last_update'] = last_time
                                
                            # 如果时间戳不新，并且距离现在超过 ResetPortTimeout = 60秒，重启进程
                            elif time.time() - last_update > ResetPortTimeout:
                                process = process_info['process']
                                print("端口",port," 长期未运行,重启该端口上的程序!\n")
                                process.terminate()
                                # 随机延时0-20s后启动
                                # 因为Simpack RT几乎不能使用并行启动/重启,需要通过随机延时避开不同端口上的同时启动
                                cmd = f"python -c \"import time, random; time.sleep(random.uniform(0, 20)); import sys; sys.path.append('{script_path}');\" && python {script_path} --port {port}"
                                new_process = subprocess.Popen(cmd, shell=True)
                                process_info['process'] = new_process
                                process_info['last_update'] = time.time()                            
        time.sleep(ResetPortTimeout/5)  # 每约10秒检查一次 

 

# 删除被多个进程占用的TCP端口        
def kill_process_on_port(port):
    # 找到占用这个端口的进程的PID
    cmd = f'lsof -t -i:{port}'
    pid_bytes = subprocess.check_output(cmd, shell=True)
    pid_str = pid_bytes.decode()  # 将字节串转换为字符串
    pids = pid_str.split()  # 分割字符串，得到每个PID
    for pid in pids:
        os.kill(int(pid), signal.SIGKILL)
                    
class SPCKenv():
    def __init__(self,port_base):
        self.port_base = port_base
        config_values = ReadConfig()
        CommPorts = AllCOMMPORTS(port_base)
        self.Maxts = config_values.Maxtimesteps
        self.CtrlIt = config_values.ControlInterval
        self.isRand = config_values.isRandomInit
        self.SumT = self.Maxts * self.CtrlIt
        self.UDP = CommPorts.UDPport_in_SPCK
        self.TCPA = CommPorts.TCPportA_2_SPCK
        self.TCPB = CommPorts.TCPportB_2_SPCK
        self.CalLogName = config_values.CalLogName
        self.StateRecordProb = 0.01 # 在一定概率下记录当前回合状态量数据
        self.is_recording = False # 是否记录数据的标志变量
       
        tcpA_in_use,tcpB_in_use = checkTCP(self.port_base)
        if(tcpA_in_use == True and tcpB_in_use == True):
            print("Python:端口",self.port_base,"的TCP-A/B均被占用,kill对应端口的SPCK")
            kill_process_on_port(self.TCPA)
            time.sleep(2)
            kill_process_on_port(self.TCPB)
            time.sleep(random.uniform(0, 20))
            OPEN_TCPA_SPCKrt(self.port_base) 
            time.sleep(random.uniform(0, 10))
            
        if(tcpA_in_use == False and tcpB_in_use == True):    
            print("Python:端口",self.port_base,"开启错误,应首先开启端口TCP-A,kill对应端口的SPCK")
            kill_process_on_port(self.TCPB)
            time.sleep(random.uniform(0, 20))
            OPEN_TCPA_SPCKrt(self.port_base) 
            time.sleep(random.uniform(0, 10))
            
        if(tcpA_in_use == False and tcpB_in_use == False):    
            print("Python:端口",self.port_base,"将自动重启 TCP-A / B")
            time.sleep(random.uniform(0, 20))
            OPEN_TCPA_SPCKrt(self.port_base) 
            time.sleep(random.uniform(0, 10))
            
        if(tcpA_in_use == True and tcpB_in_use == False):
            pass    
        
    def reset(self):
        # 在每次迭代后，写入当前的port-base和时间戳
        RecordLogFile(self.port_base,self.CalLogName)
        config_values = ReadConfig()
        self.nowsteps = 0
        # 随机化环境subvars在server端控制
        # 启动SPCK
        command = f"{config_values.Work_path}/ParallelSPCKs/Comm2SPCK {self.TCPB} {self.UDP} {self.CtrlIt} {self.SumT + 10} {config_values.Work_path} {config_values.SPCK_path}"
        pendulum_process = threading.Thread(target=run_command, args=(command,))
        pendulum_process.start() 
        time.sleep(2)  
        # 留足给SPCK启动的时间差
        
        # TCP通信初始化
        SERVER_ADDRESS = '127.0.0.1'  
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        buffer_size = 5120 
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
        self.client_socket.connect((SERVER_ADDRESS, self.TCPB))
        time.sleep(0.5) 
        # SPCK完成与python的TCP连接时需要时间
        var1, _, _, _ = ReadSubvars_ByPort(self.port_base)
        OBs = [math.sin(var1), 0, 0, 0]  
        OBs = np.array(OBs)
        # print("Python: OBs = ", OBs)
        
        # 判断本episode是否需要记录数据
        self.is_recording = random.random() < self.StateRecordProb
        self.recorded_data = [] 
        return OBs, {}
    
    def step(self,action):
        terminated = 0
        truncated = 0
        self.nowsteps = self.nowsteps + 1
        data = self.client_socket.recv(4)
        ny = struct.unpack('!I', data)[0]
        y_values = []
        for _ in range(ny):
            data = self.client_socket.recv(4)
            y_value = struct.unpack('!f', data)[0]
            y_values.append(y_value)
        OBs = [y_values[0],y_values[1], y_values[2], y_values[3]] 
        OBs = np.array(OBs)
        
        sintheta = OBs[0] # 旋转角度的sin值
        dottheta = OBs[1] # 旋转角速度
        Xcar = OBs[2]     # 车辆位移
        terminated = isTerminated(sintheta,Xcar)
        truncated = isTruncated(self.nowsteps,self.Maxts)
        # reward 修改为两个部分
        reward = CalReward_State(sintheta,dottheta,Xcar) + CalReward_Alive(terminated)
        # print("Python:  nowsteps = ", self.nowsteps)
        # print("isTerminated = ", isTerminated(RealOBs[0],RealOBs[2]))
        # print("isTruncated  = ", isTruncated(self.nowsteps,self.Maxts))  

        if  terminated or truncated: 
            # print("Python: 到达Env终止条件!")
            # print("Python: 回合总Timesteps = ", self.nowsteps)
            self.client_socket.close()
            # print("Python:关闭socket")
            
            # 如果记录数据
            if self.is_recording:
                os.makedirs('ParallelSPCKs/TrainingRecords', exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f'ParallelSPCKs/TrainingRecords/{timestamp}.txt'
                with open(filename, 'w') as f:
                    for OBs, u_values, reward in self.recorded_data:
                        OBs_str = ', '.join(map(str, OBs))
                        u_values_str = ', '.join(map(str, u_values))
                        f.write(f'OBs, {OBs_str}, u_values, {u_values_str}, reward, {reward}\n')
            time.sleep(1.5)     
            # 关闭与SPCK的socket通信
            # 此延时不可取消，否则会发生运行错误
        else:
            # 发送u变量
            action_value = action.item() if isinstance(action, np.ndarray) else action
            u_values = [action_value, 0, 0, 0]
            
            u_values_str = " ".join([f"{u_value:.4f}" for u_value in u_values])
            self.client_socket.sendall(f"{u_values_str}\n".encode('utf-8'))
            
            if self.is_recording:
                self.recorded_data.append((OBs, u_values, reward))

        return OBs, reward, terminated, truncated, {}
    
    
    

