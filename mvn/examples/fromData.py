#! /usr/bin/env python

from mvn import Mvn
from mvn.matrix import Matrix

import numpy
import pylab


def main():
    M1=Mvn.rand(2)    
    M2=Mvn.rand(2)    
    
    data1 = M1.sample(100)
    data2 = M2.sample(100)
    
    M1 = Mvn.fromData(data1)
    M2 = Mvn.fromdata(data2)
    M3 = Mvn.fromData([M1,M2])
    
    data3 = Matrix.stack([[data1],[data2]])
        
    assert M3 == Mvn.fromData(data3)

    A=pylab.gca()

    M3.plot(A,facecolor='m',minalpha=0.1,zorder = -1)
    
    M1.plot(A,facecolor='b',minalpha = 0.1,zorder = 0)
    M2.plot(A,facecolor='r',minalpha = 0.1,zorder = 1)

    pylab.scatter(data1[:,0],data1[:,1],facecolor='b',zorder = 2)
    pylab.scatter(data2[:,0],data2[:,1],facecolor='r',zorder = 3)
    

    pylab.show()

if __name__ == '__main__':
    main()
