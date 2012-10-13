#! /usr/bin/env python
import sys
import unittest
import doctest
import cPickle

def getSuite(values):
    suite=unittest.TestSuite()

    testFixture=testObjects.getObjects(values)
    
    suite.addTests(unitTests.getTests(testFixture))
    suite.addTests(docTests.getTests(mvn,testFixture))

    return suite


if __name__=='__main__':
    import mvn

    suite=unittest.TestSuite()

    if '-r' in sys.argv:
        testFixture=testObjects.testObjects
    else:
        testFixture=testObjects.makeObjects()
    
    suite.addTests(getTests(mvn,testFixture))
    


    unittest.TextTestRunner().run(suite)
