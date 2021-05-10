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

    label_dict = {}
    for root, dirs, files in os.walk("Labels/{}".format(type)) :
        for label_file in files :
            with open("{}/{}".format(root,label_file)) as lf :
                for line in lf :
                    (key, val) = line.split(',')
                    label_dict[key] = val.rstrip()
            """
            for filename in os.listdir(path) :
                if filename is not '':
                    for k,v in label_dict.items() :
                        f,e = os.path.splitext(filename)
                        if f == k :
                            new_name = "{}{}".format(v, filename)
                            os.rename("{}/{}".format(path, filename), "{}/{}".format(path, new_name))

                            move("{}/{}".format(path, new_name), "{}/classes/{}/{}".format(path, v, new_name))

                            break
            """
            for filename in os.listdir(path):
                if filename != '':

                    f, e = os.path.splitext(filename)

                    if f in label_dict :
                        v = label_dict[f]
                        new_name = "{}{}".format(v, filename)
                        os.rename("{}/{}".format(path, filename), "{}/{}".format(path, new_name))

                        move("{}/{}".format(path, new_name), "{}/classes/{}/{}".format(path, v, new_name))

if __name__ == '__main__' :
    main()