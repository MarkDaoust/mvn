#! /usr/bin/env python
import sys
import unittest
import doctest
import cPickle

import testObjects



def getTests(module,testFixture):
    jar = cPickle.dumps(testFixture)

    def setUp(test):
       test.globs.update(cPickle.loads(jar))
    
    testCases=doctest.DocTestSuite(module, setUp = setUp)
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
