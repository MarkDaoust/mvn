#! /usr/bin/env python

import mvar

import numpy
import pylab


def double():
    data1 = numpy.matrix(numpy.random.randn(100,2))*numpy.random.randn(2,2)+10*numpy.random.randn(2)
    data2 = numpy.matrix(numpy.random.randn(100,2))*numpy.random.randn(2,2)+10*numpy.random.randn(2)

    A=pylab.gca()

    M1=mvar.Mvar.fromData(data1)
    M2=mvar.Mvar.fromData(data2)

    A.add_artist(M1.patch(facecolor='b'))
    A.add_artist(M2.patch(facecolor='r'))


    pylab.scatter(data1[:,0],data1[:,1],facecolor='b')
    pylab.scatter(data2[:,0],data2[:,1],facecolor='r')

    merged = numpy.vstack([data1,data2])

    M3 = mvar.Mvar.fromData(merged)
    M4 = mvar.Mvar.fromData([M1,M2])

    assert M3 == M4

    A.add_artist(M3.patch(facecolor='m'))

    pylab.show()

if __name__ == '__main__':
    double()
