#! /usr/bin/env python

import os

outfile=open('./test_results.py','w')
outfile.write('from numpy import array,matrix\n')
outfile.write('from mvar import Mvar\n')

header=True
for line in os.popen("./mvar.py"):
    if header and line.startswith('*******'):
        header=False

    if header:
        outfile.write(line)
    else:
        outfile.write('#'+line)

os.system('git diff ./test_results.py > test_results.diff')
