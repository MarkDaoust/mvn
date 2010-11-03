#! /usr/bin/env python

import copy

class Copyable():
    def copy(self,other=None,deep=False):
        """
        either return a copy of the object, or copy another object's dictionary
        into the self. "deep" controls whether it uses copy.copy or copy.deepcopy.
        
        >>> A2=A.copy(deep=True)        
        >>> assert A2 == A
        >>> assert A2 is not A
        >>> assert A2.mean is not A.mean

        >>> A.copy(B,deep=True)
        >>> assert B == A
        >>> assert B is not A
        >>> assert A.mean is not B.mean

        set deep=False to not copy the attributes
        >>> A2=A.copy(deep=False)        
        >>> assert A2 == A
        >>> assert A2 is not A
        >>> assert A2.mean is A.mean

        >>> A.copy(B,deep=False)
        >>> assert B == A
        >>> assert B is not A
        >>> assert A.mean is B.mean
        """
         
        C=copy.deepcopy if deep else copy.copy
        if other is None:
            return C(self)
        else:
            self.__dict__.update(C(other.__dict__))