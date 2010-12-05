#! /usr/bin/env python

#builtins
import os
import sys
import doctest
import cPickle

#3rd party
import numpy

#local
import testTools

#local
import mvar
import helpers
import square
import automath    
import right
import inplace
import matrix

from matrix import Matrix
from mvar import Mvar
from right import Right

#make a dictionary of the local modules
localMods={
    'mvar'    :mvar,
    'helpers' :helpers,
    'square'  :square,
#    'automath':automath,
#    'right'   :right,
#    'inplace' :inplace,
    'matrix'  :matrix,
}


testObjects={}

if '-r' in sys.argv:
    import testObjects as TO
    testObjects=TO.__dict__()
    print "#loaded


if not testObjects:
    seed=None
    for n,item in enumerate(sys.argv):
        if item=='seed':
            seed=int(sys.argv[n+1])
            break

    testObjects=makeTestObjects(
        cplx='cplx' in sys.argv, 
        flat='flat' in sys.argv,
        seed=seed
    )
    saveTestObjects(testObjects)

for name,mod in localMods.iteritems():
    mod.__dict__.update(testObjects)
    doctest.testmod(mod)

