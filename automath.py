#! /usr/bin/env python

import itertools 
import operator

class Automath():
    """
    abctract base class
    It attempts to define a complete set of sensible operations from a more 
    limited set
    if you define (A+B) you get an inefficient positive integer multiply (A*N)
    if you define (A*B) you get an inefficient positive integer power (A*N)
    if you define mulyiply for negative integers (A*-N) you get neg (-A) and 
        sub (A-B)
    if you define power for negative integers (A**-N) you get div (A/B)
    if you define (A & B) and (A-B) and 
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
        return other*self**(-1)
    
    def __or__(self,other):
        return (self+other)-(self & other)

    def __xor__(self,other):
        return (self+other)-2*(self & other)