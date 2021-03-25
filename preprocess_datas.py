import subprocess
import os
import sys

formatters = {
    'RED' : '\033[91m',
    'GREEN': '\033[92m',
    'END': '\033[0m'
}

windows_length = sys.argv[1]
"""
try :
    print('{RED}\nGet Training/Testing Data{END}'.format(**formatters))
    subprocess.call(f'python get_datas.py', shell = True)
    print('{GREEN}Get Training/Testing Data Done\n{END}'.format(**formatters))

except Exception as identifier:
    print(identifier)
"""
try :
    print('{RED}\nCreate Label Training Data{END}'.format(**formatters))
    subprocess.call(
        f'python create_label.py -t training -l {windows_length}', shell = True)
    print('{GREEN}Create Label Training Data Done\n{END}'.format(**formatters))

    print('{RED}\nCreate Label Testing Data{END}'.format(**formatters))
    subprocess.call(
        f'python create_label.py -t testing -l {windows_length}', shell = True)
    print('{GREEN}Create Label Testing Data Done\n{END}'.format(**formatters))

except Exception as identifier:
    print(identifier)