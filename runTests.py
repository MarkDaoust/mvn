#! /usr/bin/env python
import sys

import unittest
import doctest
import pickle

import mvar

import unitTests
import docTests
import testObjects


if __name__=='__main__':
    suite=unittest.TestSuite()

    if '-r' in sys.argv:
        testFixture=testObjects.testObjects
    else:
        testFixture=testObjects.makeObjects()
    
    suite.addTests(unitTests.getTests(testFixture))
    suite.addTests(docTests.getTests(mvar,testFixture))


    unittest.TextTestRunner().run(suite)
