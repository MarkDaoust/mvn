from copyable import Copyable

def automath(cls):
    """
    class decorator:

    It attempts to define a complete set of sensible operations from a more limited set


    inspired by "total ordering"

    copy the contents of 'Automath' into the given class.
    skip entries that the class already has.
    """
    for key,value in Automath.__dict__.iteritems():
        if not hasattr(cls,key):
            setattr(cls,key,value)

    return cls
    
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

    
class Automath(Copyable,Right):        
    def __pos__(self):
        """
        >>> assert A == +A == ++A
        >>> assert A is not +A
        """
        return self.copy()

    def __neg__(self):
        """
        >>> assert -A == -1*A
        """
        return (-1)*self
    
    def __sub__(self,other):
        """
        >>> assert A-B == A+(-B)
        """
        return self+(-other)

    def __div__(self,other):
        """
        >>> assert A/B == A*B**(-1)
         """
        return self*other**(-1)

    def __ne__(self,other):
        """
        >>> assert (not (A == B)) == (A != B)  
        """
        return not (self == other)


if __debug__:
    import random

    class Test(Automath):    
        def __init__(self,num):            
            self.num = getattr(num,'num',num)

        def __int__(self):
            return int(self.num)
        
        def __float__(self):
            return float(self.num)
            
        def __str__(self):
            return '%r(%r)' % (type(self).__name__,self.num)
        
        __repr__ = __str__
        
        def __eq__(self,other):
            return self.num == getattr(other,'num',other)
            
        def __add__(self,other):
            return type(self)(self.num+getattr(other,'num',other))

        def __mul__(self,other):
            return type(self)(self.num*getattr(other,'num',other))        
        
        def __pow__(self,other):
            return type(self)(self.num**getattr(other,'num',other))
            
        def __and__(self,other):
            return type(self)(self.num+getattr(other,'num',other))

        def __or__(self,other):
            return type(self)(self.num+getattr(other,'num',other))

            
    A = Test(random.randint(-10,10))    
    B = Test(random.randint(-10,10))
    
    
if __name__ == "__main__":
    pass 