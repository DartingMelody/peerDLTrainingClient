from flask import Flask, jsonify
import requests, json, time
import urllib.request
import threading
import multiprocessing
import os
import subprocess
import shlex
from math import trunc
import argparse
import socket
import sys

def subrun(com, capture_output=False    ):
    stdout = None if not capture_output else subprocess.PIPE
    universal_newlines = None if not capture_output else True
    return subprocess.run(shlex.split(com), stdout=stdout, universal_newlines=universal_newlines, check=True)

header = {"Content-type": "application/json"}
                #  "Accept": "text/plain"} 
# ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
ip = socket.gethostname()
print(ip)
parser = argparse.ArgumentParser()
parser.add_argument('--duration', '-d', required=True, type=int)
parser.add_argument('--url', '-u', required=True)
parser.add_argument("--rank", type=int)
args = parser.parse_args()
url = args.url
duration = args.duration
print(ip, args.rank)
if(duration <= 1):
    print("duration greater than 1 minute and preferably atleast 10 minutes")
    sys.exit(1)
payload = {'spec': 'ok', 'minutes': str(duration), 'IP': ip, 'rank': args.rank}
response = requests.post(url+"register/", data = json.dumps(payload), headers=header)
user = response.json()
print(user)
start_training = False

def train(master_ip, world_size, job_id, rank):
    print(master_ip)
    print(world_size)
    print("job id is "+job_id)
    print(rank)
    start_training = True
    #resp = requests.get(url+ '/downloads/', headers=header)
    #assert(subrun('curl https://tkhaniitr-u3f2l0793la7pe20.socketxp.com/downloads/ -o outFile.zip').returncode == 0)
    dir = "../file_store/jobid"+str(job_id)+"/"
    # if int(rank) == 0:
    #     assert(subrun("unzip -o "+dir+" outFile.zip").returncode == 0)
    directory = os.path.join("../file_store/jobid/"+str(job_id)+"/")
    if not os.path.exists(directory) and int(rank) == 0:
        os.makedirs(directory)
    # run_train = 'python main.py --master-ip '+master_ip + ' --num-nodes '+str(world_size)+' --rank '+ str(rank) +' --checkpoint-dir ' + directory
    run_train = "/u/anirudhs/miniconda3/envs/py39/bin/python -m torch.distributed.launch --nnodes " + str(world_size) + " --nproc_per_node 1 --master_addr " + master_ip + " --node_rank " + str(rank) + " main.py --output_dir " + directory
    # os.chdir("/scratch/cluster/anirudhs/Code/dist/Garfield/pytorch_impl/applications/Aggregathor")
    # run_train = "/u/anirudhs/miniconda3/envs/py39/bin/python trainer.py --master " + master_ip + " --num_iter "+ str((10000//(int(world_size) - 1))) +" --dataset cifar10 --model cifarnet --batch 32 --loss cross-entropy --optimizer sgd --num_workers " + str(int(world_size) - 1) + " --fps 0 --fw 1 --gar median --rank " + str(rank)
    try:
        assert(subrun(run_train).returncode == 0)
        payload = {'user_id': user, 'job_id': job_id}
        header = {"Content-type": "application/json"}
        response = requests.post(url+"doneJob/", data = json.dumps(payload), headers=header)
        print(response)
        print("done training")
    except:
        print("exception in training thread")
        payload = {'user_id': user, 'job_id': job_id}
        header = {"Content-type": "application/json"}
        response = requests.post(url+"resetJob/", data = json.dumps(payload), headers=header)
    start_training = False
   # print("start training is "+ str(start_training))

duration = duration - 1  

while(duration > 0):
    #print(cnt)
    time.sleep(60)
    duration = duration - 1
    resp = requests.get(url+"up/"+str(user), headers=header)
    #cnt = cnt + 1
    print(resp.text)
    #print("before training "+str(start_training))
    response = resp.json()
    if  start_training is False and response['status'] == 'RUNNING': #change it to START
        start_training = True
        #print("in start training "+str(start_training))
        #_thread.start_new_thread(target = train, args = (response['master_ip'], response['world_size'],))
        thread = threading.Thread(target = train, args = (response['master_ip'], str(response['world_size'])[:-2], str(response['job_id'])[:-2],response['rank']))
        thread.start()
        # process = multiprocessing.Process(target = train, args = (response['master_ip'], str(response['world_size'])[:-2], str(response['job_id'])[:-2],response['rank']))
        # process.start()
        print('thread created')
    elif response['status'] != 'RUNNING':
        start_training = False
        print('ready to train')
