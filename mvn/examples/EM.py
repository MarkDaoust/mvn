#! /usr/bin/env python

import numpy

from mvn import Mvn
from mvn.matrix import Matrix
from mvn.mixture import Mixture

import pylab; 
pylab.ion()

source = Mixture([
    Mvn.rand(2),
    Mvn.rand(2),
])

data = source.sample(200) 


W1,R1 = [1e7],Mvn(mean=[ 10.0, 10.0],var=numpy.array([10.0,10.0])**2)
W2,R2 = [1e7],Mvn(mean=[-10.0,-10.0],var=numpy.array([10.0,10.0])**2)

old_p = numpy.inf

for N in range(10):
    pylab.gcf().clear()

    pi1 = sum(W1)
    pi2 = sum(W2)

    (pi1,pi2) = [
        pi1/(pi1+pi2),
        pi2/(pi1+pi2)
    ]

    d1 = R1.density(data)*pi1
    d2 = R2.density(data)*pi2

    (W1,W2) = [
        d1/(d1+d2),
        d2/(d1+d2),
    ]

    R1 = Mvn.fromData(data = data,weights = W1,bias=True)
    R2 = Mvn.fromData(data = data,weights = W2,bias=True)

    #print 'W1=%s' % sum(W1)
    #print 'W2=%s' % sum(W2)

    pylab.scatter(data[:,0],data[:,1],c='r',alpha=0.5, zorder = 3)
    R1.plot(zorder = 2)
    R2.plot(zorder = 1)
#    pylab.gca().add_artist(R1.patch())
#    pylab.gca().add_artist(R2.patch())
    pylab.draw()

    p=sum(pi1*W1*pi2*W2)
    print 'p=%s' % p
#    if abs(p-old_p) <0.0000001:
#        break
    old_p = p


pylab.show()