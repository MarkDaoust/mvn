#! /usr/bin/env python
"""
modified from GVR's '5 minute multimethod' class and decorator
http://www.artima.com/weblogs/viewpost.jsp?thread=101605
"""
import inspect
from trydecorate import decorator

class InCreation:
    def __init__(self,*args,**kwargs):
        raise TypeError('type InCreation cannot be instanciated')

class Multimethod(object)
def __new__(baseClass,protoFun):
        # This decorator will be wraped around the protoype function and 
        # inserted into the __call__ slot of the subclass that is used to 
        # create the returned object.
        @decorator
        def caller(protoFun,*args,**kwargs): 
            types = tuple(arg.__class__ for arg in args)
            while True:
                try:
                    found = subClass.typemap[types]
                except KeyError:
                    types = types[:-1]
                else:
                    break
            return found(*args,**kwargs)

        # match the signature of the caller, defined above, to the prototype 
        # function
        call=caller(protoFun)

        #create the subClass we'll be using 
        subClass = type(protoFun.__name__,(baseClass,),{})

        
        #make __call__ static, so the self (the multimethod) doesn't get passed
        subClass.__call__=staticmethod(call)

        #add the prototype function as the default in the typemap
        subClass.typemap={(): protoFun}
    
        #return the object created from the subclass
        return object.__new__(subClass)


    def register(self,*types):
        def register(function,types=types):                 
            if types in self.typemap:
                raise TypeError("duplicate registration")
            
            self.typemap[types] = function
            return (
                self if 
                function.__name__ == self.typemap[()].__name__ else
                function
            ) 
        return register
    

