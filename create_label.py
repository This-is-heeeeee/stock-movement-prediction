import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def main() :
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--type', help = 'the type of data', required = True)
    parser.add_argument('-l', '--seq_len', help = 'num of sequence length', default = 20)

    args = parser.parse_args()

    createLabel(args.type, args.seq_len)

def createLabel(type, seq_len) :
    if not os.path.isdir("labels"):
        os.mkdir("labels")
    print("Creating Label...")

    for root, dirs, files in os.walk("stockdatas") :
        for file in files :
            if type in file :
                removeOutput("{}/{}_label_{}.txt".format("labels",file[:-4],seq_len))
                df = pd.read_csv("{}/{}".format(root, file), parse_dates=True, index_col=0)
                df.fillna(0)

                for i in range(0, len(df)) :
                    c = df.iloc[i:i+int(seq_len)+1, :]

                    starting = 0
                    endvalue = 0
                    label = ""

                    if len(c) == int(seq_len) + 1 :
                        starting = c["Close"].iloc[-2]
                        endvalue = c["Close"].iloc[-1]

                        if endvalue > starting :
                            label = 1
                        else :
                            label = 0
                        with open("{}/{}_label_{}.txt".format("labels",file[:-4],seq_len),'a') as the_file :
                            the_file.write("{}-{},{}".format(file[:-4], i, label))
                            the_file.write("\n")

    print("Create label finished.")

def removeOutput(finput) :
    if(Path(finput)).is_file() :
        os.remove(finput)

if __name__ == '__main__' :
    main()