#! /usr/bin/env python

import itertools 
import operator

class Automath():
    """
    abctract base class
    It attempts to define a complete set of sensible operations from a more limited set
    if you define (A+B) you get an inefficient positive integer multiply (A*N)
    if you define (A*B) you get an inefficient positive integer power (A**N)
    if you define multiply for negative integers (A*-N) you get neg (-A) 
    if you define add and neg you get sub (A-B)
    if you define power for negative integers (A**-N) you get div (A/B)
    if you define and (A & B) and invert (~A) you get or
    if you define or (A|B) and sub (A-B) you get xor 
    """
    def __radd__(self,other):
        """
        assert A+B == B+A
        """
        return self+other
    
    def __rmul__(self,other):
        """
        assert A*B == B*A
        """
        return self*other
    
    def __mul__(self,N):
        """
        assert A+A+A == 3*A
        """
        return reduce(itertools.repeat(self,N),operator.add)
        
    def __pow__(self,N):
        """
        assert A**2==A*A
        """
        return reduce(itertools.repeat(self,N),operator.mul)
        
    def __pos__(self):
        """
        assert +A == A
        """
    def __neg__(self):
        """
        assert -A == -1*A
        """
        return (-1)*self
    
    def __sub__(self,other):
        """
        assert A-B == A+(-B)
        """
        return self+(-other)

    def __rsub__(self,other):
        """
        assert B-A == B+(-A)
        """
        return other+(-self)
    
    def __div__(self,other):
        """
        assert A/B == A*B**(-1)
        """
        return self*other**(-1)
        
    def __rdiv__(self,other):
        """
        assert B/A == B*A**(-1)
        """        
        return other*self**(-1)
    
    def __or__(self,other):
        """
        assert ~(A|B) == (~A&~B)
        """
        return ~(~self & ~other)

    def __xor__(self,other):
        """
        assert A^B == (A|B) & ~(A&B)
        """
        return (self|other) & ~(self & other)

    def __ne__(self,other):
        """
        assert (not (A == B)) == (A != B)  
        """
        return not (self == other)
