#! /usr/bin/env python

import __init__ as mvar
from __init__ import *

import testObjects
from testObjects import *

import numpy


A.mean = A.mean.real
A.vectors = A.vectors.real

Na = 1000

#get some data from A
Da=A.sample(Na)

#and remake the multivariates
A=Mvar.fromData(Da)

# take all the dot products
dots=(numpy.array(Da)**2).sum(1)
assert Matrix(dots) == numpy.diag(Da*Da.H)

Mean = Matrix(dots.mean())
Var = Matrix(dots.var())


assert Mean == numpy.trace(Da*Da.H)/Na
assert Mean == numpy.trace(Da.H*Da/Na)
assert Mean == (Da*Da.H).diagonal().mean()

assert A.cov+A.mean.H*A.mean == (Da.H*Da)/Na

assert Mean == numpy.trace(A.mean.H*A.mean + A.cov)
assert Mean == numpy.trace(A.mean.H*A.mean)+numpy.trace(A.cov)
assert Mean == A.mean*A.mean.H + A.trace()

#definition of variance
assert Var == (numpy.array(Mean -dots)**2).mean()

#expand it
assert Var == (Mean**2 - 2*numpy.array(Mean)*dots + dots**2 ).mean()

#distribute the calls to mean()
assert Var == Mean**2 - 2*Mean*dots.mean() + (dots**2).mean()

#but Mean == dot.mean(), so
assert Var == (dots**2).mean() - Mean**2

assert Var == (dots**2).sum()/Na - Mean**2

assert Var == ((Da*Da.H).diagonal()**2).sum()/Na - Mean**2

assert Var == Matrix((Da*Da.H).diagonal())*Matrix((Da*Da.H).diagonal()).H/Na - Mean**2

assert Mean ==(Matrix((Da*Da.H).diagonal())*Matrix.ones((Na,1))/Na)

assert Mean**2 == (Matrix((Da*Da.H).diagonal())*Matrix.ones((Na,1))/Na)**2

assert Mean**2 == (
    Matrix((Da*Da.H).diagonal()*
    Matrix.ones((Na,1))/Na) * 
    Matrix((Da*Da.H).diagonal()*
    Matrix.ones((Na,1))/Na)
)

assert Mean**2 == (
    Matrix((Da*Da.H).diagonal())*
    Matrix.ones((Na,1))*Matrix.ones((1,Na))/Na**2 * 
    Matrix((Da*Da.H).diagonal()).H
)



assert Var ==(
    Matrix((Da*Da.H).diagonal())*
    Matrix((Da*Da.H).diagonal()).H/Na 
    -
    Matrix((Da*Da.H).diagonal())*
    Matrix.ones((Na,1))*Matrix.ones((1,Na))/Na**2 * 
    Matrix((Da*Da.H).diagonal()).H
)
    
assert Var ==(
    Matrix((Da*Da.H).diagonal())*
    Matrix((Da*Da.H).diagonal()).H/Na 
    -
    (Matrix((Da*Da.H).diagonal())*Matrix((Da*Da.H).diagonal()).H.sum()).sum()/Na/Na
)

assert Var ==(
    Matrix((Da*Da.H).diagonal())/Na*
    Matrix((Da*Da.H).diagonal()).H
    -
    Matrix((Da*Da.H).diagonal())/Na*
    (numpy.trace(Da*Da.H)*Matrix.ones((Na,1)))/Na
)

assert Var == Matrix((Da*Da.H).diagonal())/Na * (
    Matrix((Da*Da.H).diagonal()).H
    -
    (numpy.trace(Da*Da.H)*Matrix.ones((Na,1)))/Na
)

assert Var == Matrix((Da*Da.H).diagonal())/Na * (
    Matrix((Da*Da.H).diagonal()).H
    -
    Mean
)
#????????????
#wiki: this is the Real value, my val
wVar=2*numpy.trace(A.cov*A.cov)+4*A.mean*A.cov*A.mean.H

assert wVar == 2*numpy.trace(
    numpy.diagflat(A.var**0.5)*
    A.vectors*A.cov*A.vectors.H*
    numpy.diagflat(A.var**0.5)+

    numpy.diagflat(A.var**0.5)*
    A.vectors*2*A.mean*A.cov*A.mean.H*A.vectors.H*
    numpy.diagflat(A.var**0.5)
    
)

assert wVar == 2*numpy.trace(
    A.cov*
    A.vectors.H*numpy.diagflat(A.var)*A.vectors
) + 4*numpy.trace(
    A.mean.H*A.mean*
    A.vectors.H*numpy.diagflat(A.var)*A.vectors*
)

assert wVar == 2*numpy.trace(
    A.cov*
    A.vectors.H*numpy.diagflat(A.var)*A.vectors
) + numpy.trace(
    4*A.mean*
    A.vectors.H*numpy.diagflat(A.var)*A.vectors*
    A.mean.H
)

assert wVar == 2*numpy.trace(
    A.cov*
    A.vectors.H*numpy.diagflat(A.var)*A.vectors
) + numpy.trace(
    4*A.mean*
    A.vectors.H*numpy.diagflat(A.var)*A.vectors*
    A.mean.H
)

assert wVar == 2*numpy.trace(A.cov*A.cov)+4*A.mean*A.cov*A.mean.H

assert wVar == 2*(A*A).trace()+4*(A*A.mean.H).trace()


