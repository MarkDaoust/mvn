#! /usr/bin/env python

##sibblings
#  #tools
#import parse
#  #tests
#import fixture
#import unit
#import doc
#    
##builtin
#import sys
#import cPickle
#import unittest
#import doctest
#
##parent module
#import mvn
#
#def main(argv):
#    parser = parse.makeParser()
#
#    (values,remainder) = parser.parse_args(argv)
#
#    assert not remainder
#
#    testFixture=fixture.getObjects(values)
#    
#    suite=unittest.TestSuite()
#    
#    suite.addTests(unit.getTests(testFixture))
#    suite.addTests(doc.getDocTests(mvn,testFixture))
#    suite.addTests(doc.getDocTests(mvn.mvncdf,testFixture))
#
#    sys.stderr.write("test values: %s\n%s\n" % (' '.join(sys.argv),values))
#        
#    unittest.TextTestRunner().run(suite)


import sys
from __init__ import main

main(sys.argv[1:])

