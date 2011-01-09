#! /usr/bin/env python
import sys

import unittest
import doctest
import pickle

import mvar

import unitTests
import docTests
import testObjects

def runTests(cplx=None,flat=None,ndim=1,seed=None):
    suite=unittest.TestSuite()

    if '-n' in sys.argv:
        testFixture=testObjects.makeObjects(cplx,flat,ndim,seed)        
    else:
        testFixture=testObjects.testObjects
    
    suite.addTests(unitTests.getTests(testFixture))
    suite.addTests(docTests.getTests(mvar,testFixture))


    unittest.TextTestRunner().run(suite)

if __name__=='__main__':
    runTests('cplx' in sys.argv,'flat' in sys.argv)

