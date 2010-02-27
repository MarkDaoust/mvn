#! /usr/bin/env python

class Inplace():
    """
    given: a class with self.copy(other) and definition of the basic forms of 
    the operators (+,-,*,/,**,&,|,^,<<,>>)

    creates inplace versions of operators
    (+=, -=, *=, /=, **=, &=, |=, ^=, <<=, >>=)
    """
    def __iadd__(self,other):
        self.copy(self + other)
        
    def __isub__(self,other):
        self.copy(self - other)

    def __imul__(self,other):
        self.copy(self * other)

    def __idiv__(self,other):
        self.copy(self / other)
        
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

