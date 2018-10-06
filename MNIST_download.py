import gzip
import urllib.request
from pathlib import Path
import os
from datetime import datetime
import sys

def process_file(url, name):
    path = os.getcwd() + '/' + name
    if Path(path).exists:
        print('Start download {} : [{}]'.format(path + '.gz', datetime.now().time()))
        urllib.request.urlretrieve(url, path + '.gz')
        print('Finish download {} : [{}]'.format(path + '.gz', datetime.now().time()))
        with gzip.open(path + '.gz', 'rb') as source, open(path, 'wb') as destination:
            print('Start unpack {} : [{}]'.format(path + '.gz', datetime.now().time()))
            destination.write(source.read())
            print('Finish unpack {} : [{}]'.format(path + '.gz', datetime.now().time()))
        os.remove(path + '.gz')

if len(sys.argv) == 1:
    sys.argv.append('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    sys.argv.append('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    sys.argv.append('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    sys.argv.append('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')

process_file(sys.argv[1], 'train-images.idx3-ubyte')
process_file(sys.argv[2], 'train-labels.idx1-ubyte')
process_file(sys.argv[3], 't10k-images.idx3-ubyte')
process_file(sys.argv[4], 't10k-labels.idx1-ubyte')