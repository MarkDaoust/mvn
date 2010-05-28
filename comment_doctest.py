#! /usr/bin/env python

import os
import sys

header=True

pipe=os.popen(' '.join(sys.argv[1:]))

broke=False

for line in pipe:
    if line.startswith('*******'):
        print '#'+line,
        broke=True
        break

    print line,

if broke:    
    for line in pipe:
        print '#'+line,
