#! /usr/bin/env python

import itertools 
import operator

class Automath():
    """
    abctract base class
    It attempts to define a complete set of sensible operations from a more 
    limited set
    if you define (+) you get an inefficient natural integer multiply (*N)
    if you define (*) you get an inefficient natural integer power (*N)
    if you define mulyiply for negative integers (*-N) you get neg (-)
        and from neg you get sub
    if you define power for negative integers (**-N) you get div (/)
    """
    def __radd__(self,other):
        return self+other
    
    def __rmul__(self,other):
        return self*other
    
    def __mul__(self,N):
        return reduce(itertools.repeat(self,N),operator.add)
        
    def __rmul__(self,N):
        return reduce(itertools.repeat(self,N),operator.add)

    def __pow__(self,N):
        return reduce(itertools.repeat(self,N),operator.pow)
        
    def __neg__(self):
        return (-1)*self
    
    def __sub__(self,other):
        return self+(-other)

    def __rsub__(self,other):
        return other+(-self)
    
    def __div__(self,other):
        return self*other**(-1)
        
    def __rdiv__(self,other):
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        return other*self**(-1)
    