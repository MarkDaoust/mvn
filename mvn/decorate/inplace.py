from copyable import copyable

def inplace(cls):
    """
    class decorator:

    add the contents of thr 'Inplace' class to another class
    
    given: a class with a definition of the basic forms of 
    the operators (+,-,*,/,**,&,|,^,<<,>>)

    creates inplace versions of operators
    (+=, -=, *=, /=, **=, &=, |=, ^=, <<=, >>=)
    """
    for key,value in Inplace.__dict__.iteritems():
        if not hasattr(cls,key):
            setattr(cls,key,value)

    return cls

@copyable
class Inplace:
    """
    this class only defines inplace operators in terms of the basic operators.
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


    

