#! /usr/bin/env python


import copy as cp

def copy(self,other=None,deep=False):
    """
    Either return a copy of the object, or copy the other object's dictionary
    into the self.

    The "deep" key word controls whether it uses copy.copy or copy.deepcopy.
    """
     
    C=cp.deepcopy if deep else cp.copy
    if other is None:
        return C(self)
    else:
        self.__dict__.update(C(other.__dict__))

def copyable(cls):
    '''
    class decorator:

    inspired by "total ordering"

    add a 'copy' method to a class
    '''
    assert not hasattr(cls,'copy')
    cls.copy=copy
    return cls

