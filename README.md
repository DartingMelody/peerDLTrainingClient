# peerDLTrainingClient
client for training ML or DL files as peer
* attach to nfsserver follow https://www.tecmint.com/install-nfs-server-on-ubuntu/ , https://www.howtoforge.com/tutorial/how-to-configure-a-nfs-server-and-mount-nfs-shares-on-ubuntu-18.04/ make mount directory to file_store
* ```pip install flask```
* Change outFile.zip to your script, it requires some argument which is required. Check outFile.zip main.py file. 
* For registering jobs -> ```curl -v -X POST -H "Content-Type: multipart/form-data" -F "title=aabb" -F "file=@outFile.zip" -F "data={\"job_duration\":\"180\",\"nodes\":\"1\"};type=application/json" https://{yoursockettocken}.socketxp.com/registerJob/ ```
* after setting up server and dispatcher, ```run client.py -d 180 -u https://{yoursockettocken}.socketxp.com/ ```  where d is client duration and u is the url for the server home

