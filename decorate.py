#! /usr/bin/env python
import sys
import types
import inspect
import functools
import itertools
import collections

from decorator import decorator

def curry(fun,*args):
    """
    @curry
    def F1(a,b,c):
        return a+b+c

    is a much nicer way of saying:

    def F1(a):
        def F2(b):
            def F3(c):
                return a+b+c
            return F3
        return F2
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
    create a decorator that runs the *args through the provided function, before 
    running the decorated function

    @prepare(lambda *args:[cleanup(arg) for arg in args])
    def doSomething(x,y,z):
        pass

    """
    @decorator
    def F(decorated,*args,**kwargs):
        return decorated(*before(*args),**kwargs)

    return F

def cleanup(after):
    """
    create a decorator that runs the output of the decorated function, through 
    the provided function

    def isImplemented(result):
        if result is NotImplemented:
            raise TypeError('NotImplemented')

    @cleanup(isImplemented)
    @MultiMethod
    def doSomething(x,y,z):
        return NotImplemented

    @doSomething.register([int,float],[int,float],[int,float])
    def doSomething(x,y,z):
        return x+y+z
    """
    @decorator
    def F(decorated,*args,**kwargs):
        return after(decorated(*args,**kwargs))

    return F

def prop(cls):
    keys = ('fget', 'fset', 'fdel')
    func_locals = {'doc':cls.__doc__}
    func_locals.update(dict((k, cls.__dict__.get(k)) for k in keys))
    return property(**func_locals)

class underConstruction(object):
    """
    class placeholder for use during multimethod creation
    """
    def __new__(cls):
        return type("underConstruction",(cls,),{})

class MultiMethod(object):
    """
    signature preserving multimethod decorator
    
    inspired by GVR's '5 minute multimethod' class and decorator
    http://www.artima.com/weblogs/viewpost.jsp?thread=101605    

    sample usage:

    Test=underConstruction('Test')

    @MultiMethod.sign(Test)
    class Test(object):
        @MultiMethod
        def __add__(self,other):
            raise TypeError('notImplemented')
        
        @__add__.register(Test):
        def addDefault(self,anything):
            raise TypeError('notImplemented')
        
        @__add__.register(Test,Test)
        def addTest(self,other):
            raise TypeError('notImplemented')

        @__add__.register(Test,dict)
        def addMapping(self,mapping):
            raise TypeError('notImplemented')

        @__add__.register(Test,[int,float])
        def addNumber(self,number):
            raise TypeError('notImplemented')

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
            raise TypeError('can only register types')

        types=itertools.product(*types)

        def register(function,types=types):
            for T in types:
                if T in self.typemap:
                    raise TypeError("duplicate registration")
            
                self.typemap[T] = function

            if function.__name__ == self.__call__.__name__:
                return self.__call__
            else:
                function.multimethod = self
                return function
        return register
    
    @staticmethod
    @curry
    def sign(signature,cls):
        """
        look for any multimethods in a class and replace the signature with 
        the class, mostly so it looks like you're talking about the class being 
        constructed
        """
        static=(
            value.__get__(())
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
            value.multimethod
            for value
            in itertools.chain(static,methods)
            if isinstance(value,types.FunctionType)
                and hasattr(value,'multimethod')
        )

        for m in multimethods:
            if isinstance(m,MultiMethod):
                typemap=m.typemap
                for key in typemap.iterkeys():
                    value=typemap.pop(key)
                    key=tuple(cls if k is signature else k for k in key)
                    typemap.update([(key,value)])

        return cls

