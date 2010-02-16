#! /usr/bin/env python

import os

print 'from numpy import array,matrix'
print 'from mvar import Mvar'

header=True
for line in os.popen("./mvar.py"):
    if header and line.startswith('*******'):
        header=False

    if header:
        print line,
    else:
        print '#'+line,
