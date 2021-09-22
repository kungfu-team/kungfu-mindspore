#!/usr/bin/env python3
import json
import os
import requests
import sys
'''
{
    "Runners": [
        {
            "IPv4": 2130706433,
            "Port": 38080
        }
    ],
    "Workers": [
        {
            "IPv4": 2130706433,
            "Port": 10000
        },
        ....
    ]
}
'''


def ipv4_to_u32(ipv4):
    a, b, c, d = [int(x) for x in ipv4.split('.')]
    return (a << 24) | (b << 16) | (c << 8) | d


def peer_id(ipv4: str, port: int):
    return {
        'IPv4': ipv4_to_u32(ipv4),
        'Port': port,
    }


port_range_start = 40000


def gen_cluster_config(n):
    runners = [peer_id('127.0.0.1', 38080)]
    workers = [peer_id('127.0.0.1', port_range_start + i) for i in range(n)]
    return {
        'Runners': runners,
        'Workers': workers,
    }


# url = 'http://127.0.0.1:9100/config'
url = 'http://127.0.0.1:9999/config'


def resize_to(n):
    config = gen_cluster_config(n)
    requests.put(url, json.dumps(config).encode())


def get_config():
    # url = os.getenv('KUNGFU_CONFIG_SERVER')
    resp = requests.get(url)
    o = resp.text
    print(o)


def main(args):
    # get_config()
    n = int(args[0])
    resize_to(n)


main(sys.argv[1:])