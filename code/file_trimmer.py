#!/usr/bin/env python
import pandas as pd
from os import scandir, makedirs
from shutil import move, rmtree
from sys import exit

def main():
    if input('WARNING: you are about to move and remove some files! Type: "Yes, I am absolutely certain of what I am doing." if you intend to continue.\n') != "Yes, I am absolutely certain of what I am doing.": exit("EXIT: You made the right decision.")
    print('Oh man...')
    
    data_path = "/home/teo/uni/Computer-Vision/data/"
    labels = pd.read_csv(data_path + "ISIC_2019_Training_GroundTruth.csv")
    
    MEL_files = labels[labels["MEL"] == 1]["image"].values
    NV_files = labels[labels["NV"] == 1]["image"].values
    
    
    makedirs(data_path + "MEL/", exist_okay=True)
    makedirs(data_path + "NV/", exist_okay=True)
    
    with scandir(data_path + "ISIC_2019_Training_Input/ISIC_2019_Training_Input/") as Dir:
        for entry in Dir:
            if (name := entry.name[:-4]) in MEL_files: move(entry.path, data_path + 'MEL/')
            elif name in NV_files: move(entry.path, data_path + 'NV/')
    
    with scandir(data_path + "MEL/") as Dir: MELdir_len = sum(1 for entry in Dir if entry.name.endswith(".jpg"))
    with scandir(data_path + "NV/") as Dir: NVdir_len = sum(1 for entry in Dir if entry.name.endswith(".jpg"))
    if MELdir_len + NVdir_len == len(MEL_files) + len(NV_files): rmtree(data_path + "ISIC_2019_Training_Input/")
    else: print('May have missed something.')


if __name__ == "__main__": main()
    