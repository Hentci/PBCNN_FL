from scapy.all import Ether, rdpcap, wrpcap

if __name__ == '__main__':
    backdoor_packets = []
    # 這個 pcap 是我們處理出來的，不是本來 demo_pcap 附的，不過實際移到 server 後應該可以改路徑讓他再跑一次
    packets = rdpcap("../data/demo_pcap/FTP-BruteForce_14-02.pcap")
    print(packets[0].mysummary)
    for packet in packets:
        # 把 reserved bit 改成 7 (111)
        packet['TCP'].reserved = 7
        # 把一些 len 跟 checksum 先清掉
        del packet['IP'].len
        del packet['IP'].chksum
        del packet['TCP'].chksum
        # 重算 len 跟 checksum
        packet = Ether(packet.build())
        backdoor_packets.append(packet)
    # 可以確認有沒有改成功
    print(backdoor_packets[0].mysummary)
    wrpcap("../data/demo_pcap/FTP-BruteForce-backdoor_14-02.pcap", backdoor_packets)