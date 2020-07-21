#!/usr/bin/env python3
# coding: utf-8
"""
    :author: pk13055
    :brief: Download bulk list of `tmdbId`s

"""
import argparse
from math import ceil
from multiprocessing import Process, Pool
import random
import sys
import time

import pandas as pd
import requests

args = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default="ids.txt",
                        help="path to file containing tmdbIds")
    parser.add_argument('-k', '--key', type=str, required=True,
                        help="tmdb API key")
    parser.add_argument('-t', '--threads', type=int, default=2,
                        help="number of threads per process")
    parser.add_argument('-p', '--process', type=int, default=4,
                        help="Number of processes to launch")
    args = parser.parse_args()
    return args


def fetch_link(id_: int, idx: int, tot: int) -> dict:
    """Fetch a tmdbId record"""
    global args
    sys.stdout.flush()

    url = f"https://api.themoviedb.org/3/movie/{id_}?api_key={args.key}"
    data = { "id": id_ }
    try:
        res = requests.get(url)
        data.update(res.json())
    except Exception as e:
        sys.stdout.write(f"[error] failed download {id_}! {e}\r\n")
    finally:
        sys.stdout.write(f"processed {id_} | [{idx} / {tot}]\r\n")
        return data


def dump_data(ids_: list):
    """Fetch data from tmdb and store as json"""
    global args
    pool = Pool(args.threads)
    start = time.time()
    res = pool.starmap_async(fetch_link,
                             [(id_, idx, len(ids_))
                              for idx, id_ in enumerate(ids_)]).get()
    df = pd.DataFrame(res)
    print(df.head())
    df.to_csv(f"{ids_[0]}-{ids_[-1]}.csv")
    print(f"Fetched {len(ids_)} records [{time.time() - start}s]")


def main():
    global args
    args = get_args()
    ids = sorted(list(map(int, open(args.filename).read().strip().splitlines())))
    random.shuffle(ids)
    n_chunks = ceil(len(ids) / args.process)
    id_chunks = [ids[i:i + n_chunks] for i in range(0, len(ids), n_chunks)]
    processess = [Process(target=dump_data, args=(id_chunk,)) for id_chunk in id_chunks]
    [_.start() for _ in processess]
    [_.join() for _ in processess]


if __name__ == "__main__":
    main()

