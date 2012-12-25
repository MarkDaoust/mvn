#! /usr/bin/env python
"""
*****************
Useful Decorators
*****************
"""

import sys
import types
import inspect
import functools
import itertools
import collections

from decorator import decorator
from automath import automath,right
from copyable import copyable
from inplace import inplace

def curry(fun,*args):
    """
    >>> @curry
    >>> def F1(a,b,c):
    >>>     return a+b+c

    is a much nicer way of saying:

    >>> def F1(a):
    >>>     def F2(b):
    >>>         def F3(c):
    >>>             return a+b+c
    >>>         return F3
    >>>     return F2
    """
    def curried(*moreArgs):
        A=args+moreArgs
        if len(A)<fun.func_code.co_argcount:
            return curry(fun,*A)
        else:
            return fun(*A)
    return curried

def prepare(before):
    """
    create a decorator that runs the positional arguments through the provided 
    function, before running the decorated function

    >>> @prepare(lambda *args:[cleanup(arg) for arg in args])
    >>> def doSomething(x,y,z):
    >>>     pass

    """
    @decorator
    def F(decorated,*args,**kwargs):
        return decorated(*before(*args),**kwargs)

    return F

def cleanup(after):
    """
    create a decorator that runs the output of the decorated function, through 
    the provided function

    >>> def isImplemented(result):
    >>>     if result is NotImplemented:
    >>>         raise TypeError('NotImplemented')
    >>>
    >>> @cleanup(isImplemented)
    >>> @MultiMethod
    >>> def doSomething(x,y,z):
    >>>     return NotImplemented
    >>>
    >>> @doSomething.register([int,float],[int,float],[int,float])
    >>> def doSomething(x,y,z):
    >>>     return x+y+z
    """
    @decorator
    def F(decorated,*args,**kwargs):
        return after(decorated(*args,**kwargs),*args,**kwargs)

    return F

def prop(cls):
    keys = ('fget', 'fset', 'fdel')
    func_locals = {'doc':cls.__doc__}
    func_locals.update(dict((k, cls.__dict__.get(k)) for k in keys))
    return property(**func_locals)

class UnderConstruction(object):
    pass

def underConstruction(name):
    """
    create a class placeholder for use in multimethod creation
    """
    return type(name,(UnderConstruction,),{})

class MultiMethod(object):
    """
    Signature preserving multimethod decorator

    inspired by GVR's `5 minute multimethods <http://www.artima.com/weblogs/viewpost.jsp?thread=101605>`_    

    sample usage:

    >>> @MultiMethod.sign(underConstruction('Test'))
    ... class Test(object):
    ...     @MultiMethod
    ...     def __add__(self,other):
    ...         raise TypeError('notImplemented')
    ...     
    ...     @__add__.register(Test):
    ...     def addDefault(self,anything):
    ...         raise TypeError('notImplemented')
    ...     
    ...     @__add__.register(Test,Test)
    ...     def addTest(self,other):
    ...         raise TypeError('notImplemented')
    ... 
    ...     @__add__.register(Test,dict)
    ...     def addMapping(self,mapping):
    ...         raise TypeError('notImplemented')
    ... 
    ...     @__add__.register(Test,[int,float])
    ...     def addNumber(self,number):
    ...         raise TypeError('notImplemented')

    """
    def __new__(baseClass,defaultFun):
        # this caller is the object that will be returned
        @decorator 
        def caller(defaultFun,*args,**kwargs): 
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
        decorated=caller(defaultFun)

        #create the subClass referenced above 
        subClass = type(defaultFun.__name__,(baseClass,),{})
        #add the prototype function as the default in the typemap
        subClass.typemap={(): defaultFun}
        #and put the decorator into the subclass
        subClass.__call__=staticmethod(decorated)
        
        #insert a multimethod object in as the register 'Method'    
        decorated.multimethod=object.__new__(subClass)
        decorated.register = decorated.multimethod.register
        
        return decorated

    def register(self,*types):
        """
        types can be individual types or sequences of types
        the sequences of types are fed through itertools.product
        before registration 
        """
        types=[
            T if isinstance(T,collections.Iterable) else (T,)
            for T in types
        ]

        if not all(type(t) is type for t in itertools.chain(*types)):
            raise TypeError('MultiMethod can only register types')

        Xtypes=itertools.product(*types)

        @curry
        def register(self,Xtypes,function):
            try:
                function = function.multimethod.last 
            except AttributeError:
                pass

            for types in Xtypes:
                if types in self.typemap:
                    raise TypeError("duplicate registration")
            
                self.typemap[types] = function

            self.last = function

            if function.__name__ == self.__call__.__name__:
                return self.__call__
            
            return function

        return register(self,Xtypes)
    
    @staticmethod
    @curry
    def sign(signature,cls):
        """
        look for any multimethods in a class and replace the signature with 
        the class, mostly so it looks like you're talking about the class being 
        constructed
        """
        static=(
            value.__func__
            for (key,value)
            in cls.__dict__.iteritems()
            if isinstance(value,(staticmethod,classmethod))
        )

        methods=(
            value
            for (key,value)
            in cls.__dict__.iteritems()
            if isinstance(value,types.FunctionType)
        )

        multimethods=(
            value.multimethod for value
            in itertools.chain(static,methods)
            if isinstance(value,types.FunctionType)
                and hasattr(value,'multimethod')
        )

        for m in multimethods:
            if isinstance(m,MultiMethod):
                replace = dict(
                    (
                        tuple(
                            cls if T is signature else T 
                            for T in key
                        ) if signature in key else key
                        ,value
                    )
                    for key,value in m.typemap.iteritems() 
                )
                m.typemap.clear()
                m.typemap.update(replace)

        return cls

