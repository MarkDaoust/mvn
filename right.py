#! /usr/bin/env python

def right(cls):
    """
    class decorator:

    inspired by "total ordering"

    copy the contents of 'Right' into the given class.
    skip entries that the class already has.
    """
    for key,value in Right.__dict__.iteritems():
        if not hasattr(cls,key):
            setattr(cls,key,value)

    return cls

class Right():
    """
    This class just creates the right half of many operators that 
    can be easily calculated from the left versions and a few simple 
    operations
    """
    def __radd__(self,other):
        """
        >>> assert A+B == B+A
        """
        return self+other

    def __rmul__(self,other):
        """
        >>> assert A*B == B*A
        """
        return self*other

    def __rsub__(self,other):
        """
        >>> assert B-A == B+(-A)
        """
        return other+(-self)
    
    def __rdiv__(self,other):
        """
        >>> assert B/A == B*A**(-1)
        """        
        return other*self**(-1)

    def __rand__(self,other):
        """
        >>> assert A & B == B & A
        """
        return self & other

    def __ror__(self,other):
        """
        >>> assert A | B == B | A
        """
        return other | self
  
    def __rxor__(self,other):
        """
        >>> assert A ^ B == B ^ A
        """
        return other ^ self


