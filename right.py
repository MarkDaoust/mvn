#! /usr/bin/env python

class Right(object):
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
