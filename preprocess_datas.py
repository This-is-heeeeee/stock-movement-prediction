import subprocess
import os
import sys

formatters = {
    'RED' : '\033[91m',
    'GREEN': '\033[92m',
    'END': '\033[0m'
}

windows_length = sys.argv[1]
dimension = sys.argv[2]
work_type = sys.argv[3]

try :
    print('{RED}\nGet Training/Testing Data{END}'.format(**formatters))
    subprocess.call(f'python get_datas.py -l {windows_length} -t {work_type}', shell = True)
    print('{GREEN}Get Training/Testing Data Done\n{END}'.format(**formatters))

except Exception as identifier:
    print(identifier)

try :
    print('{RED}\nCreate Label Data{END}'.format(**formatters))
    subprocess.call(f'python create_label.py -t {work_type} -l {windows_length}', shell = True)
    print('{GREEN}Create Label Data Done\n{END}'.format(**formatters))


except Exception as identifier:
    print(identifier)

try :
    print('{RED}\nConvert Data to Candlestick img{END}'.format(**formatters))
    subprocess.call(f'python data_to_img.py -l {windows_length} -d {dimension} -t {work_type}', shell=True)
    print('{GREEN}Convert Data to Candlestick img Done\n{END}'.format(**formatters))

except Exception as identifier:
    print(identifier)

try :
    print('{RED}\nLabelling Dataset{END}'.format(**formatters))
    subprocess.call(f'python img_to_dataset.py -i dataset/{windows_length}_{dimension} -t {work_type}', shell=True)
    print('{GREEN}CLabelling Dataset Done\n{END}'.format(**formatters))

except Exception as identifier:
    print(identifier)
