#! /usr/bin/env python
"""
inspired by GVR's '5 minute multimethod' class and decorator
http://www.artima.com/weblogs/viewpost.jsp?thread=101605
"""
import types
import inspect
from decorator import decorator

def sign(cls):
    """
    look for any multimethods in a class and replace 
    """
    multimethods=set(
        value.multimethod
        for (key,value) 
        in cls.__dict__.iteritems() 
        if isinstance(value,types.FunctionType) and hasattr(value,'multimethod')
    )

    for m in multimethods:
        typemap=m.typemap
        for key in typemap.iterkeys():
            value=typemap.pop(key)
            key=tuple(cls if k is None else k for k in key)
            typemap.update([(key,value)])

    return cls

class MultiMethod(object):
    """
    signature preserving multimethod decorator
    """
    def __new__(baseClass,protoFun):
        # this caller is the object that will be returned
        @decorator 
        def caller(protoFun,*args,**kwargs): 
            types = tuple(arg.__class__ for arg in args)
            while True:
                try:
                    found = subClass.typemap[types]
                except KeyError:
                    #pop types until you find a match
                    types = types[:-1]
                else:
                    break
            return found(*args,**kwargs)

        # we use the decorator module to match the signature to the prototype
        decorated=caller(protoFun)

        #create the subClass referenced above 
        subClass = type(protoFun.__name__,(baseClass,),{})
        #add the prototype function as the default in the typemap
        subClass.typemap={(): protoFun}
        #and put the decorator into the subclass
        subClass.__call__=staticmethod(decorated)
        
        #insert a multimethod object in as the register 'Method'    
        decorated.multimethod=object.__new__(subClass)
        decorated.register=decorated.multimethod.register
        return decorated


    def register(self,*types):
        def register(function,types=types):                 
            if types in self.typemap:
                raise TypeError("duplicate registration")
            
            self.typemap[types] = function

            if function.__name__ == self.__call__.__name__:
                return self.__call__
            else:
                function.multimethod = self
                return function
        return register
    

