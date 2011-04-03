#! /usr/bin/env python

import __init__ as mvar
from __init__ import *

import testObjects
from testObjects import *

import numpy


A.mean = A.mean.real
A.vectors = A.vectors.real

B.mean = B.mean.real
B.vectors = B.vectors.real

Na = 200
Nb = 100

N=Na*Nb

#get some data from A and B
Da=A.sample(Na)
Db=B.sample(Nb)

#and remake the multivariates
A=Mvar.fromData(Da)
B=Mvar.fromData(Db)

# take all*all the dot product
dot=numpy.array(Da*Db.H)

#the population mean
Mean = Matrix(dot.mean())
#the population variance
Var = Matrix(dot.var())

#should equal the distribution mean
assert Mean == A.mean*B.mean.H

#definition of variance
assert Var == (numpy.array(Mean -dot)**2).mean()

#expand it
assert Var == (Mean**2 - 2*numpy.array(Mean)*dot + dot**2 ).mean()

#diftribute the calls to mean()
assert Var == Mean**2 - 2*Mean*dot.mean() + (dot**2).mean()

#but Mean == dot.mean(), so
assert Var == (dot**2).mean() - Mean**2

dot = Matrix(dot)

assert Var == numpy.trace(dot*dot.H)/N - Mean**2
#factor everything
assert Var == numpy.trace(Da.H*Db*Db.H*Da)/Na/Nb - (A.mean*B.mean.H)**2


#rotate the trace
assert Var == numpy.trace(Da.H*Da*Db.H*Db)/Na/Nb - (A.mean*B.mean.H)**2

#group the data's
assert Var == numpy.trace((Da.H*Da)*(Db.H*Db))/Na/Nb - (A.mean*B.mean.H)**2

#distribute the N's
assert Var == numpy.trace((Da.H*Da)/Na*(Db.H*Db)/Nb) - (A.mean*B.mean.H)**2

#from the definition of mean and cov
assert A.cov+A.mean.H*A.mean == (Da.H*Da)/Na
assert B.cov+B.mean.H*B.mean == (Db.H*Db)/Nb

#replace
assert Var == numpy.trace((A.cov+A.mean.H*A.mean)*(B.cov+B.mean.H*B.mean))-(A.mean*B.mean.H)**2


#multiply it out
assert Var == numpy.trace(A.cov*B.cov + A.mean.H*A.mean*B.cov + A.cov*B.mean.H*B.mean + A.mean.H*A.mean*B.mean.H*B.mean) - (A.mean*B.mean.H)**2

#distribute the calls to tracea

assert Var == (
    numpy.trace(A.cov*B.cov) + 
    numpy.trace(A.mean.H*A.mean*B.cov) + 
    numpy.trace(A.cov*B.mean.H*B.mean) +
    numpy.trace(A.mean.H*A.mean*B.mean.H*B.mean)
) - (A.mean*B.mean.H)**2

#rotate traces
assert Var == (
    numpy.trace(A.cov*B.cov) + 
    numpy.trace(A.mean*B.cov*A.mean.H) + 
    numpy.trace(B.mean*A.cov*B.mean.H) +
    numpy.trace(A.mean*B.mean.H*B.mean*A.mean.H)
) - (A.mean*B.mean.H)**2

#remove traces for scalars
assert Var == (
    numpy.trace(A.cov*B.cov) + 
    A.mean*B.cov*A.mean.H + 
    B.mean*A.cov*B.mean.H +
    (A.mean*B.mean.H)*(B.mean*A.mean.H)
) - (A.mean*B.mean.H)**2

#cancel means
assert Var == numpy.trace(A.cov*B.cov) + A.mean*B.cov*A.mean.H + B.mean*A.cov*B.mean.H

#avoid covariance matrixes
assert Var == (A*B).trace() + (B*A.mean.H).trace() + (A*B.mean.H).trace()

