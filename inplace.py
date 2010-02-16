#! /usr/bin/env python

class Inplace():
    """
    given: self.copy(other)
    produces: +=, -=, *=, /=, **=, &=, |=, ^=, <<=, >>=
    
    attempts creates inplace versions of operators
    """
    def __iadd__(self,other):
        self.copy(self+other)
        
    def __isub__(self,other):
        self.copy(self-other)

    def __imul__(self,other):
        self.copy(self*other)

    def __idiv__(self,other):
        self.copy(self/other)
        
    def __iand__(self,other):
        self.copy(self & other)
    
    def __ior__(self,other):
        self.copy(self | other)

    def __ixor__(self,other):
        self.copy(self ^ other)
    
    def __ilshift__(self,other):
        self.copy(self << other)
        
    def __irshift__(self,other):
        self.copy(self >> other)

