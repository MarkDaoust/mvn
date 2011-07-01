#! /usr/bin/env python
import sys
import optparse
import unittest
import doctest
import pickle

import __init__ as mvar

import testObjects

import docTests
import unitTests

def makeParser():
    parser=optparse.OptionParser()

    parser.add_option('-x',action='store_true',help = "cross test using various values of lfatness and datatype")

    new=optparse.OptionGroup(parser,'new')    
    new.add_option('-n','--new',action='store_true',default=False,help='create new test objects')
    parser.add_option_group(new)

    general=optparse.OptionGroup(parser,'general')
    general.add_option('-s','--seed',action='store',type=int,help='set the random seed')
    general.add_option('-d','--ndim',action='store',type=int,help='set the number of data dimensions')
    parser.add_option_group(general)

    datatype=optparse.OptionGroup(parser,'DataType')
    datatype.add_option('-r','--real',dest='dtype',action='store_const',const=('r','r','r'), help='use real data')
    datatype.add_option('-c','--cplx',dest='dtype',action='store_const',const=('c','c','c'), help='use complex data')
    datatype.add_option('-i','--imag',dest='dtype',action='store_const',const=('i','i','i'), help='use imaginary data')
    datatype.add_option('--type',dest='dtype',nargs=3,type=str,help='set data types individually to the test objects')
    parser.add_option_group(datatype)

    flatness=optparse.OptionGroup(parser,'Flatness')
    flatness.add_option('-f','--flat',dest='flat',action='store_const',const=(True,True,True),help='set all the objects to flat')
    flatness.add_option('-F','--full',dest='flat',action='store_const',const=(False,False,False),help='set no objects to flat')
    flatness.add_option('--flatness',nargs=3,dest='flat',type=int,help='set flatness individually')
    parser.add_option_group(flatness)

    return parser

def getSuite(values):
    suite=unittest.TestSuite()

    testFixture=testObjects.getObjects(values)
    
    suite.addTests(unitTests.getTests(testFixture))
    suite.addTests(docTests.getTests(mvar,testFixture))

    return suite

if __name__=='__main__':
    parser = makeParser()

    (values,remainder) = parser.parse_args()

    assert not remainder

    suite=unittest.TestSuite()

    if values.x:
        for flatness in [(True,True,True),(False,False)]:
            for dtype in ['r','c','i']:
                values.flatness=flatness
                values.dtype=dtype
                suite.addTests(getSuite(values))
    else:
        suite.addTests(getSuite(values))


    unittest.TextTestRunner().run(suite)
