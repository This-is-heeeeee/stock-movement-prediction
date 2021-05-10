import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import mplfinance as mpf
import os
import argparse

def main() :
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--seq_len', help='num of sequence length', default=20)
    parser.add_argument('-d', '--dimension', help='a dimension value', type=int, default=48)
    parser.add_argument('-t', '--type', help = 'the type of data', required = True)
    args = parser.parse_args()

    makeCandlechart(args.seq_len, args.dimension, args.type)

def makeCandlechart(seq_len, dimension, type) :
    if not os.path.exists('dataset/{}_{}/{}'.format(seq_len, dimension,type)) :
        os.makedirs('dataset/{}_{}/{}'.format(seq_len, dimension,type))

    mc = mpf.make_marketcolors(up='tab:red', down='tab:blue', edge='inherit')
    myStyle = mpf.make_mpf_style(marketcolors=mc)
    plt.style.use('dark_background')

    for root, dirs, files in os.walk("stockdatas"):
        for file in files :
            if type in file :
                df = pd.read_csv("{}/{}".format(root, file), parse_dates=True, index_col=0)
                df.fillna(0)

                for i in range(0, len(df)) :
                    c = df.iloc[i:i+int(seq_len), :]
                    if len(c) == int (seq_len) :
                        imgfile = 'dataset/{}_{}/{}/{}-{}.png'.format(seq_len, dimension, type, file[:-4], i)
                        mydpi = 96

                        fig = plt.figure(figsize=(dimension/mydpi, dimension/mydpi), dpi=mydpi)
                        spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1])
                        ax1 = fig.add_subplot(spec[0])
                        ax1.grid(False)
                        #ax1.set_xticklabels([])
                        #ax1.set_yticklabels([])
                        ax1.xaxis.set_visible(False)
                        ax1.yaxis.set_visible(False)
                        ax1.axis('off')

                        ax2 = fig.add_subplot(spec[1])
                        ax2.grid(False)
                        #ax2.set_xticklabels([])
                        #ax2.set_yticklabels([])
                        ax2.xaxis.set_visible(False)
                        ax2.yaxis.set_visible(False)
                        ax2.axis('off')

                        mpf.plot(c, type='candle', ax=ax1, volume=ax2, style=myStyle)
                        fig.savefig(imgfile, pad_inches=0, transparent=False)
                        plt.close(fig)

if __name__ == "__main__" :
    main()
