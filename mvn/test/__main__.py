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

def getSuite(values):
    suite=unittest.TestSuite()

    testFixture=fixture.getObjects(values)
    
    suite.addTests(unit.getTests(testFixture))
    suite.addTests(getDocTests(mvn,testFixture))

    return suite


def getDocTests(module,testFixture):
    jar = cPickle.dumps(testFixture)

    def setUp(test):
       test.globs.update(cPickle.loads(jar))
    
    testCases=doctest.DocTestSuite(module, setUp = setUp)
    return testCases
    

def main(argv):
    parser = parse.makeParser()

    (values,remainder) = parser.parse_args(argv)

    assert not remainder

    suite=unittest.TestSuite()

    if values.x:
        for flatness in [(True,True,True),(False,False,False)]:
            values.flatness=flatness
            suite.addTests(getSuite(values))
    else:
        suite.addTests(getSuite(values))

    suite.addTests(getDocTests(mvn.mvncdf,fixture.getObjects(values)))

    sys.stderr.write("test values: %s\n%s\n" % (' '.join(sys.argv),values))
        
    unittest.TextTestRunner().run(suite)



if __name__ == "__main__":
    main(sys.argv[1:])
