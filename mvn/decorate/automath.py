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
#  
#    def __rxor__(self,other):
#        """
#        >>> assert A ^ B == B ^ A
#        """
#        return other ^ self


    
class Automath(Copyable,Right):
    """
    don't repeat yourself
    
    this class fills in a few obvious operators,
    """

#    """
#    if you define (A+B) you get an inefficient positive integer multiply (A*N)
#    if you define (A*B) you get an inefficient positive integer power (A**N)
#    if you define multiply for negative integers (A*-N) you get neg (-A) 
#    if you define add and neg you get sub (A-B)
#    if you define power for negative integers (A**-N) you get div (A/B)
#    if you define div and floordiv (A//B) you get divmod (A%B)
#    if you define and (A & B) and invert (~A) you get or (A|B)
#    if you define or (A|B) and sub (A-B) you get xor (A^B) 
#    if you define eq (A=B) and gt (A>B)  you'll get lt (A<B), le (A<=B) ,ne (A!=b) and bool (A!=0)
#    if you define rshift you get lshift    
#    """
    
#    def __mul__(self,N):
#        """
#        Use __add__ and __sub__ to create __mul__
#        
#        >>> assert 3*A == A+A+A
#        >>> assert 0*A == A-A
#        >>> assert (-N)*A == A-(N+1)*A
#        """
#        assert int(N) == N
#        N = int(N)        
#
#        if N>0:
#            return reduce(operator.add,itertools.repeat(self,N))
#
#        # block infinite recursion
#        assert self.__sub__.im_func is not Automath.__sub__.im_func
#        
#        if N==0:
#            return self-self
#
#        assert N<0            
#        return self-(N+1)*self
#            
#        
#    def __pow__(self,N):
#        """
#        use __mul__ and __div__ to produce __pow__
#        
#        >>> assert M**2==M*M
#        >>> assert M**0 == M/M
#        >>> assert M**(-N) == M/(M**(N+1))
#        """
#        assert int(N) == N
#        N = int(N)
#
#        if N>0:
#            return reduce(operator.mul,itertools.repeat(self,N))
#
#        # block infinite recursion
#        assert self.__div__.im_func is not Automath.__div__.im_func
#
#        if N == 0:
#            return self/self
#            
#        assert N<0
#        return self/(self**(abs(N)+1))

        
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

#    def __mod__(self,other):
#        """        
#        >>> from 
#        >>> assert A%B == A/B - A//B
#        """
#        return (self.__truediv__(other)-self.__flordiv__(other))*other
#    
#    def __or__(self,other):
#        """
#        >>> assert (A|B) == ~(~A&~B)
#        """
#        return ~(~self & ~other)
#
#    def __xor__(self,other):
#        """
#        >>> assert A^B == (A|B) & ~(A&B)
#        """
#        return (self|other) & ~(self & other)

    def __ne__(self,other):
        """
        >>> assert (not (A == B)) == (A != B)  
        """
        return not (self == other)

#    def __ge__(self,other):
#        """
#        >>> assert (A>=B) == (A==B or A>B)
#        """
#        return self==other or self>other
#
#    def __lt__(self,other):
#         """
#         >>> assert (A<B) == 1-(A>=B)
#         """
#         return 1 - self>=other
#
#   def __le__(self,other):
#        """
#        >>> assert A>=B) == (A==B + A>B)
#        """
#        return 1 - self>other
#
#    def __lshift__(self,other):
#        """
#        >>> assert (A << B) == (A >> -B)
#        """
#        return A >> -B




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
            return '%s(%s)' % (type(self).__name__,self.num)
        
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