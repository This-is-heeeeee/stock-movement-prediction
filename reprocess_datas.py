import os
import argparse
from shutil import move

def main() :
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', help='a csv file of stock data', required=True)
    parser.add_argument('-t', '--type', help='the type of data', required=True)
    args = parser.parse_args()

    img_to_dataset(args.input, args.type)

def img_to_dataset(input, type) :
    path = "{}/{}".format(input, type)
    folders = ['1', '0']

    if not os.path.exists("{}/classes".format(path)):
        os.mkdir("{}/classes".format(path))
    for folder in folders :
        if not os.path.exists("{}/classes/{}".format(path,folder)):
            os.mkdir("{}/classes/{}".format(path,folder))

    for root, dirs, files in os.walk("{}/classes/".format(path)) :
        for labeled_img in files :
            if labeled_img != '':
                new_name = labeled_img[1:]
                os.rename("{}/{}".format(root,labeled_img), "{}/{}".format(root,new_name))
                move("{}/{}".format(root, new_name), "{}/{}".format(path, new_name))

if __name__ == '__main__' :
    main()