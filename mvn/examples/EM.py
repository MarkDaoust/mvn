#! /usr/bin/env python

import numpy

from mvn import Mvn
from mvn.matrix import Matrix
from mvn.mixture import Mixture

import pylab; pylab.ion()

#source = Mixture(
#    distributions=[
#        Mvn.fromData(Matrix.randn([500,2])*(Matrix.eye(2)+Matrix.randn([2,2]))),
#        Mvn.fromData(Matrix.randn([500,2])*(Matrix.eye(2)+Matrix.randn([2,2]))),    
#    ],
#    weights=[numpy.random.rand(),numpy.random.rand()],
#)
#data = soure.sample(200) 

D1 = Mvn.rand().sample(1000)
#Matrix.randn([1000,2])*(Matrix.eye(2)+Matrix.randn([2,2]))
D2 = Mvn.rand().sample(1000)
#Matrix.randn([100,2])*(Matrix.eye(2)+Matrix.randn([2,2]))    


M1 = Mvn.fromData(D1)
M2 = Mvn.fromData(D2)

print 'M1=%s' % M1
print 'M2=%s' % M2

data = numpy.vstack([
    D1,
    D2,
])


W1,R1 = [1e7],Mvn(mean=[ 10.0, 10.0],var=numpy.array([20.0,20.0])**2)
W2,R2 = [1e7],Mvn(mean=[-10.0,-10.0],var=numpy.array([20.0,20.0])**2)

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

    pylab.scatter(data[:,0],data[:,1],c='r',alpha=0.5)
    pylab.gca().add_artist(R1.patch())
    pylab.gca().add_artist(R2.patch())
    pylab.draw()

    p=sum(pi1*W1*pi2*W2)
    print 'p=%s' % p
#    if abs(p-old_p) <0.0000001:
#        break
    old_p = p


pylab.show()