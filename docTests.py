#! /usr/bin/env python
import sys
import unittest
import doctest

import testObjects

def getTests(module,testFixture):
    module.__dict__.update(testFixture)
    testCases=doctest.DocTestSuite(module)
    return testCases
    

if __name__=='__main__':
    import mvar

    suite=unittest.TestSuite()

    if '-r' in sys.argv:
        testFixture=testObjects.testObjects
    else:
        testFixture=testObjects.makeObjects()
    
    suite.addTests(getTests(mvar,testFixture))


    unittest.TextTestRunner().run(suite)
