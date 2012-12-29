#! /usr/bin/env python
"""
****************
Plane Base Class
****************
"""
import numpy

import mvn.helpers as helpers
import mvn.decorate as decorate
from mvn.matrix import Matrix 
from mvn.square import square


Plane = decorate.underConstruction('Plane')

@decorate.MultiMethod.sign(Plane)
class Plane(decorate.Automath):
    rtol = 1e-5
    """
    relative tolerence
    
    see :py:func:`mvn.helpers.approx`
    """
    
    atol = 1e-8
    """
    absolute tolerence
    
    see :py:func:`mvn.helpers.approx`
    """    
    
    
    """
    plane class, meant to (eventually) factor out some code, and utility from the Mvn class
    """
    def __init__(
        self,
        vectors=Matrix.eye,
        mean=numpy.zeros,
    ):
        mean= mean if callable(mean) else numpy.array(mean).flatten()[None,:]
        vectors= vectors if callable(vectors) else Matrix(vectors)

        stack=helpers.autoshape([
            [vectors],
            [mean   ],
        ],default=1)
        
        #unpack the stack into the object's parameters
        self.vectors = Matrix(numpy.real_if_close(stack[0,0]))
        self.mean    = Matrix(numpy.real_if_close(stack[1,0]))

    def __repr__(self):
        """
        print self
        """
        return '\n'.join([
            '%s(' % self.__class__.__name__,
            '    mean=',
           ('        %r,' % self.mean).replace('\n','\n'+8*' '),
            '    vectors=',
           ('        %r' % self.vectors).replace('\n','\n'+8*' '),
            ')',
        ])

    __str__ = __repr__
    
    def __getitem__(self,index):
        """
        project the plane into the selected dimensions
        """
        assert not isinstance(index,tuple),'1-dimensional index only'
        
        return type(self)(
            mean=self.mean[:,index],
            vectors=self.vectors[:,index],
        )


    @decorate.prop
    class shape():
        """
        get the shape of the vectors,the first element is the number of 
        vectors, the second is their lengths: the number of dimensions of 
        the space they are embedded in
            
        >>> assert A.vectors.shape == A.shape
        >>> assert (A.vectors.shape[0],A.mean.size)==A.shape
        >>> assert A.shape[0]==A.rank
        >>> assert A.shape[1]==A.ndim
        """
        def fget(self):
            return self.vectors.shape
            
    @decorate.prop
    class rank():
        """
        get the number of dimensions of the space covered by the mvn
        
        >>> assert A.rank == A.vectors.shape[0]
        """
        def fget(self):
            return self.vectors.shape[0]

    @decorate.prop
    class ndim(object):
        """
        get the number of dimensions of the space the mvn exists in
        
        >>> assert A.ndim==A.mean.size==A.mean.shape[1]
        >>> assert A.ndim==A.vectors.shape[1]
        """
        def fget(self):
            return self.mean.size
            
    @decorate.prop
    class flat(object):
        """
        >>> assert bool(A.flat) == bool(A.vectors.shape[1] > A.vectors.shape[0]) 
        """
        def fget(self):
            return max(self.vectors.shape[1] - self.vectors.shape[0],0)
            
    def __nonzero__(self):
        """
        True if not empty
        
        >>> assert A
        >>> assert bool(A) == bool(A.ndim)
        >>> assert not A[:0]
        """
        return bool(self.ndim)

    @decorate.MultiMethod
    def __add__(self,other):
        """
        add two planes together
        """
        raise TypeError("No Apropriate Method Found")

    @__add__.register(Plane)
    def __add__(self,other):
        result = self.copy()
        result.mean = result.mean+other
        return result

    @__add__.register(Plane,Plane)
    def __add__(self,other):
        return Plane(
            mean=self.mean+other.mean,
            vectors=numpy.vstack([self.vectors,other.vectors])
        )

    def getNull(self,vectors = None):
        if vectors is None:
            vectors = self.vectors
        
        shape=vectors.shape        
        missing = shape[1]-shape[0]
    
        if missing>0:
            vectors = numpy.vstack(
                [vectors,numpy.zeros((missing,shape[1]))]
            )
        else:
            vectors=vectors
    
        var,vectors = square(vectors)
    
        zeros=self.approx(var)
    
        return vectors[zeros]
        
    def approx(self,*args):
        return helpers.approx(*args,atol = self.atol,rtol = self.rtol)


    def __and__(self,other):
        """
        plane intersection
        """
        Nself=self.getNull()
        Nother=other.getNull()

        #and stack them
        null=numpy.vstack([
            Nself,
            Nother,
        ])

        #get length of the component of the means along each null vector
        r=numpy.vstack([Nself*self.mean.H,Nother*other.mean.H])
        
        mean = (numpy.linalg.pinv(null,1e-6)*r).H

        return type(self)(vectors=self.getNull(null),mean=mean)


if __debug__:
    ndim = helpers.randint(1,10)
    ndim2 = helpers.randint(1,ndim)    
    
    A = Plane(
        mean = numpy.random.randn(1,ndim),
        vectors = numpy.random.randn(ndim2,ndim)
    )