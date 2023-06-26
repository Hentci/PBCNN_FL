import time
import os
import sys
import pymongo

sys.path.append('/home/sage/PBCNN/config/')
import Attacker_ips

from tools import tool2, tool3
from scapy.all import PcapReader

# from tool2
def save_str_mongo(pcap_file, database_name = 'PacketInString_backdoor', reset = False, timeout = 64, timeout2 = 120, file_buffer = 20):
    if os.path.isdir(pcap_file):
        label_time = pcap_file.split(os.sep)[-1]
        pcap_paths = [os.path.join(pcap_file, x) for x in sorted(os.listdir(pcap_file))]
    else:
        label_time = pcap_file.split(os.sep)[-1][:-5]
        pcap_paths = [pcap_file]

    label, day_mon = label_time.split('_')
    day, mon = day_mon.split('-')
    a_ips = getattr(Attacker_ips, f'attacker_ips_{day}_{mon}')
    # 把 label 設為 benign
    label = 'benign'

    collection_name = label
    mongo_session = pymongo.MongoClient()
    mongo_db = mongo_session.get_database(database_name)
    if reset and collection_name in mongo_db.list_collection_names():
        mongo_db.drop_collection(collection_name)
    mongo_col = mongo_db.get_collection(collection_name)

    flows_maps = dict()
    cnt = 0
    pro_wr = 0
    for i, path in enumerate(pcap_paths):
        if i > 0 and i % file_buffer == 0:
            tmp = []
            for f in list(flows_maps.values()):
                f['byte_len'] = sum([len(x) for x in f['pkts_list']]) // 2
                cnt += 1
                tmp.append(tool2._too_large_help(f))
            print(f'Writing {len(tmp)} to mongo .')
            mongo_col.insert_many(tmp)
            flows_maps.clear()
            del tmp
        print(f'===> Handle on {path} ')
        with PcapReader(path) as pr:
            for pkt in pr:
                # if inet.IP not in pkt.layers():
                if not pkt.haslayer('IP'):
                    continue
                ip_layer = pkt.getlayer("IP")
                if ip_layer.src not in a_ips and ip_layer.dst not in a_ips:
                    continue
                bid = tool2.get_biflow_id(pkt)
                if isinstance(bid, Exception):
                    pro_wr += 1
                    continue

                biflow_id, protocol = bid
                pkt_str = tool2.pkt_to_str(pkt)

                if biflow_id in flows_maps:
                    cur_biflow = flows_maps[biflow_id]
                    last_seen_time = cur_biflow['last_seen_time']

                    if int(float(pkt.time) - cur_biflow['begin_time']) >= timeout2 \
                            or int(float(pkt.time) - last_seen_time) >= timeout:
                        cur_biflow['is_finished'] = True
                        cur_biflow['byte_len'] = sum([len(x) for x in cur_biflow['pkts_list']]) // 2
                        mongo_col.insert_one(tool2._too_large_help(cur_biflow))
                        cnt += 1
                        flows_maps[biflow_id] = {
                            'biflow_id': biflow_id,
                            'pkts_list': [pkt_str],
                            'begin_time': float(pkt.time),
                            'last_seen_time': float(pkt.time),
                            'is_finished': False,
                            'label': label
                        }
                    else:
                        cur_biflow['pkts_list'].append(pkt_str)
                        cur_biflow['last_seen_time'] = float(pkt.time)
                else:
                    flows_maps[biflow_id] = {
                        'biflow_id': biflow_id,
                        'pkts_list': [pkt_str],
                        'begin_time': float(pkt.time),
                        'last_seen_time': float(pkt.time),
                        'is_finished': False,
                        'label': label
                    }
    
    for ele in list(flows_maps.values()):
        ele['byte_len'] = sum([len(x) for x in ele['pkts_list']]) // 2
        mongo_col.insert_one(tool2._too_large_help(ele))
        cnt += 1
    print(f"Biflow cnt: {cnt} , wrong protocol: {pro_wr}, DONE !!!")
    mongo_session.close()

# from tool3
def final_data(new_db='backdoor_613', raw_db='PacketInString_backdoor'):
    train_ids, valid_ids, test_ids = [], [], []

    client = pymongo.MongoClient()
    raw_db = client.get_database(raw_db)
    new_db = client.get_database(new_db)

    print(f'==> shuffling by cols...')
    col_names = raw_db.list_collection_names()
    for name in col_names:
        col = raw_db.get_collection(name)
        cur_ids = []
        for bs in col.find(no_cursor_timeout=True):
            cur_ids.append((name, bs['_id']))
        
        cur_ids = tool3._shuffle(cur_ids)
        len1 = int(len(cur_ids) * 0.6)
        len2 = int(len(cur_ids) * 0.7)

        train_ids.extend(cur_ids[:len1])
        valid_ids.extend(cur_ids[len1:len2])
        test_ids.extend(cur_ids[len2:])
        # 這邊有把 undersample 跟 oversample 拿掉
    
    print(' ===== train ===== ')
    train_ids = tool3._shuffle(train_ids)
    train_col = new_db.get_collection('train')
    tool3._insert_by_id(train_col, train_ids, raw_db)

    print(' ===== validation ===== ')
    valid_ids = tool3._shuffle(valid_ids)
    valid_col = new_db.get_collection('valid')
    tool3._insert_by_id(valid_col, valid_ids, raw_db)

    print(' ===== test ===== ')
    test_ids = tool3._shuffle(test_ids)
    test_col = new_db.get_collection('test')
    tool3._insert_by_id(test_col, test_ids, raw_db)


def pcap_to_str_mongo():
    print("#### backdoor pcap ---> mongo ####")
    save_str_mongo(pcap_file = 'FTP-BruteForce-backdoor_14-02.pcap', database_name = 'PacketInString_backdoor')
    print("all backdoor done.")

if __name__ == '__main__':
    s = time.time()
    print("Start backdoor preprocess...")
    pcap_to_str_mongo()
    final_data(new_db='backdoor_613', raw_db='PacketInString_backdoor')

    tool3.str_mongo_to_sparse_tfrecord(save_dir='/trainingData/sage/CIC-IDS2018/tfrecord_backdoor/', col_name='train', db_name='backdoor_613', bs=10000)
    tool3.str_mongo_to_sparse_tfrecord(save_dir='/trainingData/sage/CIC-IDS2018/tfrecord_backdoor/', col_name='test', db_name='backdoor_613', bs=10000)
    tool3.str_mongo_to_sparse_tfrecord(save_dir='/trainingData/sage/CIC-IDS2018/tfrecord_backdoor/', col_name='valid', db_name='backdoor_613', bs=10000)

    print("backdoor preprocess done!!!")
    print(f'cost: {(time.time() - s) // 60} min')