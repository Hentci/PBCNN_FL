from tools.tool2 import save_benign_mongo
import time

import os
import sys
import time

from tools import tool2
from tools.tool3 import shuffle_by_col_and_mixed, str_mongo_to_sparse_tfrecord, oversample, final_data

def pcap_to_str_mongo():
    print("#### attack pcap ---> mongo ####")
    print("processing...")
    files = [
             '/trainingData/sage/CIC-IDS2018/attack/Bot_02-03.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/Brute-Force-Web_22-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/Brute-Force-Web_23-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/Brute-Force-XSS_22-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/Brute-Force-XSS_23-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DDoS-attacks-LOIC-HTTP_20-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DDOS-HOIC_21-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DDOS-LOIC-UDP-1_21-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DDOS-LOIC-UDP-2_21-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DDoS-LOIC-UDP_20-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DoS-GoldenEye_15-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DoS-Hulk_16-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DoS-SlowHTTPTest_16-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/DoS-Slowloris_15-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/FTP-BruteForce_14-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/Infiltration_01-03.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/Infiltration_28-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/SQL-Injection_22-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/SQL-Injection_23-02.pcap',
             '/trainingData/sage/CIC-IDS2018/attack/SSH-Bruteforce_14-02.pcap'
             ]
    for fi in files:
        tool2.save_str_mongo(pcap_file=fi,
                             database_name='PacketInString')

    # 有事先把一樣 label 的檔案合起來了，應該不會用到這個
    '''
    dirs = [
        './data_cache/raw_attacker_packets/dos-hulk_16-02',
        './data_cache/raw_attacker_packets/ssh-bruteforce_14-02',
        './data_cache/raw_attacker_packets/ddos-loic-http_20-02',
        './data_cache/raw_attacker_packets/ddos-hoic_21-02'
        './data_cache/raw_attacker_packets/ddos-loic-udp_20-02',
        './data_cache/raw_attacker_packets/ddos-loic-udp_21-02'
    ]

    for di in dirs:
        tool2.save_str_mongo(pcap_file=di,
                             database_name='PacketInString')
    '''

    print("all attack done.")

def main():
    print("begin preprocessing...")
    # process benign pcap files and save into mogodb
    ''' DONE!!
    save_benign_mongo(pcap_dir='/trainingData/sage/CIC-IDS2018/benign/',
                      label='benign',
                      database_name='PacketInString')    
    '''

    
    '''
    save_benign_mongo(pcap_dir='/home/fgtc/Documents/notebooks/data_cache/raw_benign_packets2',
                      label='benign2',
                      database_name='PacketInString')
    '''
    # process attack pcap files and save into Mongodb
    # Done!!
    # pcap_to_str_mongo()

    # split train, valid, test and save to Mongodb
    final_data(oversample_dict={'infiltration': 10,
                                'bruteforce-web': 10,
                                'bruteforce-xss': 20,
                                'sql-injection': 40},
               undersample_dict={'ddos-hoic': 4},
               new_db='mixed_613',
               raw_db='PacketInString')
    # convert train, valid, test to tfrecord.
    str_mongo_to_sparse_tfrecord(save_dir='/trainingData/sage/CIC-IDS2018/tfrecord/',
                                 col_name='train', db_name='mixed_str', bs=10000)

    str_mongo_to_sparse_tfrecord(save_dir='/trainingData/sage/CIC-IDS2018/tfrecord/',
                                 col_name='test', db_name='mixed_str', bs=10000)

    str_mongo_to_sparse_tfrecord(save_dir='/trainingData/sage/CIC-IDS2018/tfrecord/',
                                 col_name='valid', db_name='mixed_str', bs=10000)

    print("preprocess done!!!")

if __name__ == '__main__':
    s = time.time()
    main()
    print(f'cost: {(time.time() - s) // 60} min')

