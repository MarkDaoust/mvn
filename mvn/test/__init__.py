import os

import nose

import doc
import fixture
import parse
import unit

def main(argv):
    [dirname,filename] = os.path.split(__file__) 
    [dirname,_] = os.path.split(dirname)
    
    targets = [dirname,dirname+'/test/unit.py']
    nose.run(argv = ['-v6','--with-doctest']+targets)

