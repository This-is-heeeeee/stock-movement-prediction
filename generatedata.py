import os
import sys
from shutil import copyfile

pathdir = sys.argv[1]
origindir = sys.argv[2]
targetdir = sys.argv[3]
work = sys.argv[4]


def createDir(pathdir, targetdir):
    if not os.path.exists("{}/{}".format(pathdir, targetdir)):
        os.mkdir("{}/{}".format(pathdir, targetdir))

    if not os.path.exists("{}/{}/{}".format(pathdir, targetdir, work)):
        os.mkdir("{}/{}/{}".format(pathdir, targetdir, work))

    if not os.path.exists("{}/{}/{}/0".format(pathdir, targetdir, work)):
        os.mkdir("{}/{}/{}/0".format(pathdir, targetdir, work))

    if not os.path.exists("{}/{}/{}/1".format(pathdir, targetdir, work)):
        os.mkdir("{}/{}/{}/1".format(pathdir, targetdir, work))


createDir(pathdir, targetdir)

counttest = 0
counttrain = 0

for root, dirs, files in os.walk("{}/{}/{}/classes".format(pathdir, origindir, work)):
    for file in files:
        if file[0] == '0':
            origin = "{}/{}".format(root, file)
            destination = "{}/{}/{}/0/{}".format(pathdir, targetdir, work, file)
            copyfile(origin, destination)
            counttrain += 1

        elif file[0] == '1':
            origin = "{}/{}".format(root, file)
            destination = "{}/{}/{}/1/{}".format(pathdir, targetdir, work, file)
            copyfile(origin, destination)
            counttrain += 1


print(counttrain)
print(counttest)