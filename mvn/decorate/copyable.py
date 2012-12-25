import copy

def copyable(cls):
    '''
    class decorator:

    inspired by "total ordering"

    add a 'copy' method to a class
    '''
    assert not hasattr(cls,'copy')
    cls.copy=Copyable.__dict__['copy']
    return cls



class Copyable(object):
    def copy(self,other=None,deep=False):
        """
        Either return a copy of the object, or copy the other object's dictionary
        into the self.
    
        The "deep" key word controls whether it uses copy.copy or copy.deepcopy.
        """
         
        C=copy.deepcopy if deep else copy.copy
        if other is None:
            return C(self)
        else:
            self.__dict__.update(C(other.__dict__))
