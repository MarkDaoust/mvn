#! /usr/bin/env python

#builtin
import sys
import cPickle
import unittest
import doctest

#parent module
import mvn

#sibblings
  #tools
import load
import parse
  #tests
import fixture
import unit

def getDocTests(module,testFixture):
    def setUp(test,jar = cPickle.dumps(testFixture)):
       test.globs.update(cPickle.loads(jar))
    
    testCases=doctest.DocTestSuite(module, setUp = setUp)
    return testCases
    

def main(argv):
    parser = parse.makeParser()

    (values,remainder) = parser.parse_args(argv)

    assert not remainder

    testFixture=fixture.getObjects(values)
    
    suite=unittest.TestSuite()
    
    suite.addTests(unit.getTests(testFixture))
    suite.addTests(getDocTests(mvn,testFixture))
    suite.addTests(getDocTests(mvn.mvncdf,testFixture))

    sys.stderr.write("test values: %s\n%s\n" % (' '.join(sys.argv),values))
        
    unittest.TextTestRunner().run(suite)



if __name__ == "__main__":
    main(sys.argv[1:])
