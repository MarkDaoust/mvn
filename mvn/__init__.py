#! /usr/bin/env python

#todo: better interoperability with scipy.stats
#todo: mixtures and or "|" operator -> pymix? 
#todo: try open-bayes? 
#todo: try sympy's version of this (sympy.stats)?
#todo: transform's in the "&" operator
#todo: find a way to auto update the multimethod call dictionaries when sub-classes are created.
#todo: find the standard approach to making things plottable in matplotlib
#todo: try mayavi
#todo: finish the "transposed" and and "Complex" subclasses   
#todo: cleanup the multimethods so there are no complicated call diagrams
#todo: revert-Mvn mul --> Matrix*Mvn == Mvn.fromData(Matrix)*Mvn
#todo: add a dimension preserving "marginal" function,  
#todo: formData should be the main constructor
#todo: merge chain and  sensor.measure?
#todo: type handling- be more uniform, arrays work element wize, matrixes get converted to Mvns ?
#todo: better type handling, multimethods? many things that accept an mvn should 
#        accept Mvn.eye, Mvn.zeros, Mvn.infs 
#todo: relay the canonization step, I think the default can (should?) be none
#todo: add cholsky decomposition
#todo: wikipedia/kalmanfiltering#information filter    !!   the mvn and it's 
#       inverse are different things, maybe the linear algebra should go in a 
#       super class, and all the covariance/information filter stuff in two sub classes 
#todo: Exceptions,error handling
#todo: do something about mvns with zero dimensions ?
#todo: understand transforms composed of Mvns as the component vectors, and 
#          whether it is meaningful to consider mvns in both the rows and columns
#todo: implement transpose and dot product, in relation to quadratic and bilinear forms ? 
#todo: chi2 distribution/non-central chi2 for the lengths (other distributions?)
#todo: cleanup the 'square' function (now that it is clear that it's half of an SVD)
#todo: understand the relationship between these and a hessian matrix.
#todo: figure out the relationship between these and spherical harmonics
#todo: investigate higher order cumulants, 'principal cumulant analysis??' <- or not http://www.johndcook.com/blog/2010/09/20/skewness-andkurtosis/


"""
*********************************
Multivariate Normal Distributions
*********************************

`Multivariate Normal Distributions <http://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_ 
packaged to act like a `vector <http://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_. 

The goal is to make these objects work as intuitively as possible, to make algorithms 
like `Kalman Filtering <http://en.wikipedia.org/wiki/Kalman_filter>`_, 
`Principal Component Analysis <http://en.wikipedia.org/wiki/Principal_component_analysis>`_, 
and `Expectation Maximization <http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_ 
easy to implement, and understand.

The documentation is full of examples. The objects used in the examples are 
created by :py:func:`mvn.testObjects.makeObjects`, and stored in 
:py:mod:`mvn.testObjects`. 

In all the documentation examples:
    | **ndim** is the number of dimensions of the example objects
    | **A**, **B** and **C** are instances of the Mvn class
    | **K1** and **K2** are random real numbers
    | **M** and **M2** are matrixes
    | **E** is an apropriately sized eye Matrix
    | **N** is an integer

But remember that circular logic works because circluar logic works. A lot of 
the examples are demonstrations of what the code is doing, or expected invariants. 
They don't prove I'm right, but only that I'm being consistant.
"""
############  imports

## builtin
import functools

## 3rd party
import numpy
numpy.seterr(all = 'ignore')

import scipy
import pylab
import matplotlib
import matplotlib.lines
import matplotlib.patches

## local
import helpers
from helpers import sqrt
from square import square

from matrix import Matrix
from mixture import Mixture

#decorations
import decorate

#base class
from plane import Plane

import mvncdf

__all__ = ['Mvn']

Mvn=decorate.underConstruction('Mvn')
Mvn.T=decorate.underConstruction('Mvn.T')

@decorate.MultiMethod.sign(Mvn)
class Mvn(Plane):
    """  
    .. inheritance-diagram:: mvn.Mvn
    
    Principal References:
        | http://www.autonlab.org/tutorials/gaussian.html 
        | http://en.wikipedia.org/wiki/Kalman_filter
        | http://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/
 
    Basic math operators (+,-,*,/,**,&,|) have been overloaded to work as 
    consistantly as possible. There are several surprising features in the math 
    these things produce, so watchout.  

    The & operator does a baysian inference update (like the kalman filter 
    update step).
    
    >>> posterior = prior & evidence #doctest: +SKIP

    This considerably simplifies some manipulations, kalman filtering, 
    for example becomes: 
            
    >>> state[t+1] = (state[t]*STM + noise) & measurment #doctest: +SKIP
        
    Sensor fusion, for uncorrelated sensors reduces to:    
            
    >>> result = measurment1 & measurment2 & measurment3 #doctest: +SKIP
        
    
    Attributes:
        | **mean** : mean of the distribution
        | **var** :  the variance asociated with each vector
        | **vectors** : unit eigen-vectors, as rows
        
    """ 

    infoBase = numpy.e
    """
    default base to use in formation calculations
    
    >>> assert Mvn.infoBase is numpy.e
    """    

    rtol = 1e-5
    """
    relative tolerence
    
    see :py:meth:`mvn.Mvn.squeeze`
    """
    
    atol = 1e-8
    """
    absolute tolerence
    
    see :py:meth:`mvn.Mvn.squeeze`
    """


    ############## Creation
    def __init__(
        self,
        vectors=Matrix.eye,
        var=Matrix.ones,
        mean=Matrix.zeros,
        **kwargs
    ):
        """
        Create an Mvn from the mean and decomposed covariance matrix
        
        >>> assert numpy.multiply(A.vectors.H,A.var)*A.vectors == A.cov

        all parameters should be (real valued and array like) or (callable with 
        a size tuple as the only argument)
        
        :param vectors: *shape=(N,M)*, like eigenvectors but doesn't need to be a unitary matrix
        :param var: *shape=(M,)*, like eigenvalues, defaults to :py:meth:`Matrix.ones`, 
        :param mean: *shape=(1,N)*, Mean of the distribution
        :param ** kwargs: key words are retransmitted to :py:meth:`mvn.Mvn.cannonize`
        
        set 'square' to false if you know your vectors already form a unitary 
        matrix. set 'squeeze' to false if you don't want small variances, less
        than :py:attr:`mvn.Mvn.atol`, to  automatically removed
        
        
        """
        #stack everything to check sizes and automatically inflate any 
        #functions that were passed in
        
        if callable(var):
            default = 0
        else:
            var = numpy.array(var).flatten()[:,None]
            default = var.size

        var= var if callable(var) else numpy.array(var).flatten()[:,None]
        mean= mean if callable(mean) else numpy.array(mean).flatten()[None,:]
        vectors= vectors if callable(vectors) else Matrix(vectors)
        
#todo split 'autostack' into 'autoexpand' and 'stack'
        stack=Matrix(helpers.autostack([
            [var,vectors],
            [1  ,mean   ],
        ],default = default))

        #unpack the stack into the object's parameters
        self.mean = stack[-1,1:]
        self.var = numpy.array(stack[:-1,0]).flatten()
        self.vectors = stack[:-1,1:]
        
        assert  (numpy.isreal(numpy.asarray(self.mean)).all() 
            and numpy.isreal(numpy.asarray(self.var)).all()
            and numpy.isreal(numpy.asarray(self.vectors)).all()
        ),'real numbers only'

        self.cannonize(**kwargs)
        
    def cannonize(self,square=True,squeeze=True):
        """
        called at the end if :py:meth:`mvn.Mvn.__init__`, used to put the object 
        into a cannonical form
        """
        if square:
            self.copy(self.square())

        if squeeze:
            self.copy(self.squeeze())

    ############## alternate creation methods
    @classmethod
    def format(cls,something):
        '''
        take an arraylike object and return a :py:class:`mvn.matrix.Matrix` (for 
        2d data), a :py:func:`numpy.array` (for Nd data), or the unmodified object 
        (for 0d data)
        
        >>> assert Mvn.format(A) is A
        >>> assert isinstance(Mvn.format([1,2,3]),numpy.ndarray)
        >>> assert isinstance(Mvn.format([[1,2,3]]),Matrix)
        >>> assert isinstance(Mvn.format([[[2]]]),numpy.ndarray)
        '''
        A=numpy.array(something)
                     
        if A.ndim == 2:
            something=numpy.asmatrix(A)
            something.__class__=Matrix
        elif A.ndim != 0:
            something=A
        
        return something
    
    @classmethod
    def sum(cls,data,weights=None):
        """
        :param data:
        :param weights:
        """
        N = cls._getN(data,weights)
        self = cls.fromData(data,weights)
        return self*N

    @classmethod
    def mean(cls,data,weights=None):
        """
        :param data:
        :param weights:
        """
        N = cls._getN(data,weights)
        return cls.sum(data,weights)*[1.0/N]
    
    @classmethod
    def _getN(cls,data,weights):
        """
        :param data:
        :param weights:
        """
        return (
            data.shape[0] 
            if weights is None 
            else numpy.sum(weights)
        )
    
    @classmethod
    def _getWeights(cls,weights,data,N):
        """
        :param data:
        :param weights:        
        """        
        return (
            numpy.ones(data.shape[0])
            if weights is None 
            else numpy.array(weights)
        )/float(N)
    
    @classmethod
    def _getMean(cls,data,mean,weights):
        """
        :param data:
        :param weights:        
        """
        if mean is None:
            mean = numpy.multiply(weights[:,None],data).sum(0)
        elif callable(mean):
            mean = mean(data.shape[1])
    
        mean = numpy.asmatrix(mean)
    
        return mean
    
    
    @classmethod    
    @decorate.prepare(lambda cls,data,mean:[cls,cls.format(data),cls.format(mean)])
    @decorate.MultiMethod
    def fromData(cls,data,mean=None,**kwargs):
        """
        :param data:  
        :param mean:
        :param ** kwargs:
        
        optional arguments:
            mean - array like, (1,ndim) 
            weights - array like, (N,) 
            bias - bool
    
            anything else is passed through to from cov
    
        >>> assert Mvn.fromData(A)==A 
    
        >>> assert Mvn.fromData([1,2,3]).ndim == 1
    
        >>> data = [[1,2,3]]
        >>> new=Mvn.fromData(data)
        >>> assert new.mean == data
        >>> assert Matrix(new.var) == Matrix.zeros
        >>> assert new.vectors == Matrix.zeros
        >>> assert new.cov == Matrix.zeros
        
        bias is passed to numpy's cov function.
        
        any kwargs are just passed on the Mvn constructor.
        
        this creates an Mvn with the same mean and covariance as the supplied 
        data with each row being a sample and each column being a dimenson
        
        remember numpy's default covariance calculation divides by (n-1) not 
        (n) set bias = false to use n-1,
        """
        return cls.fromMatrix(Matrix(data),**kwargs)
        
    fit = fromData
    
    @classmethod
    @fromData.__func__.register(type,Mvn,type(None))
    def fromMvn(cls,self,mean=None):
        """
        :param self:  
        :param mean:
            
        >>> assert Mvn.fromData(A)==A
        """
        return self.copy(deep = True)
    
    @classmethod    
    @fromData.__func__.register(type,Mvn)
    def fromMvnOffset(cls,self,mean=Matrix.zeros):
        """
        :param self:  
        :param mean:
            
        think paralell axis theorem
        
        >>> a=A[:,0]
        >>> assert Mvn.fromData(a,mean=0).mean == Matrix.zeros
        >>> assert Mvn.fromData(a,mean=0).cov == a.cov+a.mean.H*a.mean
    
        >>> assert Mvn.fromData(A,mean=Matrix.zeros).mean == Matrix.zeros
        >>> assert Mvn.fromData(A,mean=Matrix.zeros).cov == A.cov+A.mean.H*A.mean
        """
        if callable(mean):
            mean=mean(self.ndim)
    
        delta=(self-mean)
    
        vectors = delta.mean
    
        subVectors=delta.vectors 
        subWeights=delta.var
         
        return cls(
            mean = mean,
            var = numpy.concatenate([[1],subWeights]),
            vectors = numpy.concatenate([vectors,subVectors]),
        )
    
    @classmethod
    @fromData.__func__.register(type,numpy.ndarray)
    def fromArray(cls,data,mean=None,weights=None,bias=True):
        """
        :param data:  
        :param mean:
        :param weights:
        :param bias:
        
        >>> data1 = numpy.random.randn(100,2)+5*numpy.random.randn(1,2)
        >>> data2 = numpy.random.randn(100,2)+5*numpy.random.randn(1,2)
        >>>
        >>> mvn1 = Mvn.fromData(data1)
        >>> mvn2 = Mvn.fromData(data2)
        >>>
        >>> assert Mvn.fromData([mvn1,mvn2]) == Mvn.fromData(numpy.vstack([data1,data2]))
    
        >>> N1=1000
        >>> N2=10
        >>> data1 = numpy.random.randn(N1,2)+5*numpy.random.randn(1,2)
        >>> data2 = numpy.random.randn(N2,2)+5*numpy.random.randn(1,2)
        >>>
        >>> mvn1 = Mvn.fromData(data1)
        >>> mvn2 = Mvn.fromData(data2)
        >>>
        >>> assert Mvn.fromData([mvn1,mvn2],weights=[N1,N2]) == Mvn.fromData(numpy.vstack([data1,data2]))
        """
        if data.dtype is not numpy.dtype('object'):
            return cls.fromMatrix(Matrix(data).T,weights=weights,mean=mean,bias=bias)
    
        ismvn=numpy.array([isinstance(vector,Mvn) for vector in data])    
        mvns=data[ismvn]
    
        data=numpy.array([
            numpy.squeeze(vector.mean if mvn else vector)
            for mvn,vector 
            in zip(ismvn,data)
        ])
    
        N=cls._getN(data,weights)-(not bias)
        weights=cls._getWeights(weights,data,N)
        mean = cls._getMean(data,mean,weights)
    
        subVectors=numpy.vstack([
            mvn.vectors 
            for mvn in mvns
        ])
    
        subWeights=numpy.concatenate([
            w*mvn.var
            for w,mvn in zip(weights[ismvn],mvns)
        ])
    
        vectors=data-numpy.array(mean)
    
        return cls(
            mean = mean,
            var = numpy.concatenate([weights,subWeights]),
            vectors = numpy.concatenate([vectors,subVectors]),
        )
    
    @classmethod
    @fromData.__func__.register(type,Matrix)
    def fromMatrix(cls,data,mean=None,weights=None,bias=True,**kwargs):
        """
        :param data:
        :param mean:
        :param weights:
        :param bias:
            
        >>> D=Mvn.fromData([[0],[2]])
        >>> assert D.mean == 1
        >>> assert D.var == 1
    
        >>> D=Mvn.fromData([[0],[2]],mean=[0])
        >>> assert D.mean == 0
        >>> assert D.var == 2
    
        """        
        N = cls._getN(data,weights)-(0 if bias else 1)
        
        if bias and not N:
            return cls.infs(data.shape[1])
            
        weights = cls._getWeights(weights,data,N)
        mean = cls._getMean(data,mean,weights)
    
        vectors=data-mean
    
        return cls(
            mean=mean,
            var=weights,
            vectors=vectors,
        )
    
    @classmethod
    def fromCov(cls,cov,**kwargs):
        """
        :param cov:
        :param ** kwargs:
            
        everything in kwargs is passed directly to the constructor
        """
        cov=Matrix(cov)

        diag = Matrix(numpy.diag(cov))
        eig = numpy.linalg.eigh if abs(diag) == diag else numpy.linalg.eig
        #get the variances and vectors.
        (var,vectors) = eig(cov) if cov.size else (Matrix.zeros([0,1]),Matrix.zeros([0,0]))
        vectors=Matrix(vectors.H)     

        return cls(
            vectors=vectors,
            var=var,
            square=False,
            **kwargs
        )


    @classmethod
    def zeros(cls,n=1,mean=Matrix.zeros):
        """
        :param n:
        :param mean:
            
        >>> n=abs(N)
        >>> Z=Mvn.zeros(n)
        >>> assert Z.mean==Matrix.zeros
        >>> assert Z.var.size==0
        >>> assert Z.vectors.size==0
        >>> assert Z**-1 == Mvn.infs
        """
        if callable(mean):
            mean=mean([1,n])

        return cls(mean=mean)
    
    @classmethod
    def infs(cls,n=1,mean=None):
        """
        :param n:
        :param mean:
            
        >>> n=abs(N)
        >>> inf=Mvn.infs(n)
        >>> assert inf.mean==Matrix.zeros
        >>> assert inf.var.size==inf.mean.size==n
        >>> assert Matrix(inf.var)==Matrix.infs
        >>> assert inf.vectors==Matrix.eye
        >>> assert inf**-1 == Mvn.zeros
        """
        result = cls.zeros(n)**-1
        if mean is not None:
            result.mean = mean
        return result

    @classmethod
    def eye(cls,n=1,mean = None):
        """
        :param n:
        :param mean:
        
        >>> n=abs(N)
        >>> eye=Mvn.eye(n)
        >>> assert eye.mean==Matrix.zeros
        >>> assert eye.var.size==eye.mean.size==n
        >>> assert Matrix(eye.var)==Matrix.ones
        >>> assert eye.vectors==Matrix.eye
        >>> assert eye**-1 == eye
        """
        return cls(
            mean=Matrix.zeros([1,n]) if mean is None else mean,
            vectors=Matrix.eye(n),
        )
        
    @classmethod
    def rand(cls,shape = 2):
        """
        :param shape:
            
        generate a random multivariate-normal distribution
        (just for testing purposes, no theoretical basis)
        
        >> M = Mvn.rand(A.shape)
        >> assert M.shape = A.shape
            
        .. plot:: ../examples/rand.py main

        """
        if hasattr(shape,'__iter__'):
            height,ndims = shape
        else:
            height,ndims = shape,shape   
        
        randn = Matrix.randn
        eye= Matrix.eye
        
        return cls.fromData(
            randn([height+1,ndims])*
            (eye(ndims)+randn([ndims,ndims])/3)+
            3*randn()*randn([1,ndims])
        )

#something is wrong here
    @decorate.MultiMethod
    def diag(self,**kwargs):
        """
        default multimethod is only reachable with python3
        """
        return Mvn(var = self,**kwargs)
    
    @diag.register(Mvn)
    def diag(self):
        """
        Return a distribution with the same marginals, but zero correlation 
        between elements.
        
        this is a maximum entropy distribution, 
        if all you want is to preserve the marginal variances

        >>> assert A.entropy() <= A.diag().entropy()
        >>> assert A.diag().corr == Matrix.eye
        >>> assert sorted(A.diag().var) == sorted(A.width()**2)
        >>>
        >>> marginals = [A[:,dim] for dim in range(A.ndim)]
        >>> assert Mvn.stack(*marginals) == A.diag()
        """
        return type(self)(mean=self.mean,var = self.width()**2)   
    
    
    ##### 'cosmetic' manipulations
    def inflate(self):
        """
        add the zero length direction vectors to make the matrix square

        >>> if A.shape[0] == A.shape[1]:
        ...     assert A*A.vectors.H*A.vectors==A
        
        >>> if A.shape[0] != A.shape[1]:
        ...     assert A*A.vectors.H*A.vectors!=A

        >>> A=A.inflate()
        >>> assert A*A.vectors.H*A.vectors==A        
        """
        result = self.copy()

        shape=self.shape        

        missing = self.flat

        if missing == 0:
            return result
        elif missing<0:
            result = result.square()

        result.vectors = numpy.vstack(
            [self.vectors,numpy.zeros((missing,shape[1]))]
        )

        result.var = numpy.concatenate([result.var,numpy.zeros(missing)])
        
        result = result.square()

        zeros=helpers.approx(result.var)

        result.var[zeros]=0

        return result

    def squeeze(self):
        """
        drop any vector/variance pairs where the variance is out of tolerence
        (:py:attr:`mvn.Mvn.rtol`, :py:attr:`mvn.Mvn.atol`, :py:func:`helpers.approx`)

        >>> assert A.inflate().squeeze().shape == A.shape
        """
        result=self.copy()
#        small=helpers.approx(self.var,rtol = self.rtol, atol = self.atol)
        small=helpers.approx(self.var)
        
        if small.size:
            result.var = result.var[~small]
            result.vectors = result.vectors[~small,:]
        
        return result

    def square(self):
        """
        squares up the vectors, so that the 'vectors' matrix is unitary 
        (rotation matrix extended to complex numbers)
        
        >>> assert A.vectors*A.vectors.H==Matrix.eye
        """ 
        result = self.copy()
        (result.var,result.vectors)=square(
            vectors=result.vectors,
            var=result.var,
        )

        return result

    
    ############ setters/getters -> properties

    @decorate.prop
    class cov(object):
        """
        get or set the covariance matrix
        
        >>> assert A.cov == numpy.multiply(A.vectors.H,A.var)*A.vectors
        >>> assert abs(A).cov == A.scaled.H*A.scaled
        
        >>> a = A.copy()
        >>> a.cov = B.cov
        >>> assert a.cov == B.cov
        
        >>> a = A.copy()
        >>> #implicit covariance extraction
        >>> a.cov = B
        >>> assert a.cov == B.cov        
        """
        def fget(self):
            return numpy.multiply(self.vectors.H,self.var)*self.vectors

        def fset(self,cov):
            if isinstance(cov,Mvn):
                self.var = cov.var
                self.vectors = cov.vectors
            else:
                new=type(self).fromCov(
                    mean=self.mean,
                    cov=cov,
                )
                
                self.copy(new)

    @decorate.prop
    class corr(object):
        """
        get or set the correlation matrix used by the object
        
        >>> assert A.corr==(A/A.width()).cov
        
        >>> a = A.copy()
        >>> a.corr = B.corr
        >>> assert Matrix(a.width()) == A.width()
        >>> assert Matrix(a.mean) == A.mean
        >>> assert Matrix(a.corr) == B.corr
        
        >>> a = A.copy()
        >>> # implicit extracton of correlation matrix
        >>> a.corr = B
        >>> assert Matrix(a.width()) == A.width()
        >>> assert Matrix(a.mean) == A.mean
        >>> assert Matrix(a.corr) == B.corr

        
        >>> a = A.copy()
        >>> a.corr = A.corr
        >>> assert a == A
        """
        def fget(self):
            return (self/self.width()).cov
          
        def fset(self,corr):
            
            if isinstance(corr,Mvn):
                corr = corr/corr.width()
                mean = self.mean
                self.copy(corr*self.width())
                self.mean = mean
            else:
                new = Mvn.fromCov(corr)*self.width()
                new.mean = self.mean
                self.copy(new)
            

    @decorate.prop
    class scaled(object):
        """
        get the vectors, scaled by the standard deviations. 
        Useful for transforming from unit-eigen-space, to data-space

        >>> assert A.vectors.H*A.scaled==A.transform()
        """
        def fget(self):
            return Matrix(numpy.multiply(sqrt(self.var[:,None]),self.vectors))
        
    
    @decorate.prop
    class flat(object):
        """
        >>> assert bool(A.flat) == bool(A.vectors.shape[1] > A.vectors.shape[0]) 
        """
        def fget(self):
            return max(self.vectors.shape[1] - self.vectors.shape[0],0)

    @decorate.prop
    class ndim(object):
        """
        get the number of dimensions of the space the mvn exists in
        
        >>> assert A.ndim==A.mean.size==A.mean.shape[1]
        >>> assert A.ndim==A.vectors.shape[1]
        """
        def fget(self):
            return self.mean.size
            
    def __nonzero__(self):
        """
        True if not empty
        
        >>> assert A
        >>> assert bool(A.ndim) == bool(A.ndim)
        >>> assert A[:0]
        >>> assert not A[:,:0]
        """
        return bool(self.ndim)

    @decorate.prop
    class rank():
        """
        get the number of dimensions of the space covered by the mvn
        
        >>> assert A.rank == A.var.size
        >>> assert A.rank == A.vectors.shape[0]
        """
        def fget(self):
            return self.vectors.shape[0]

    def _transformParts(self,power=1):
        """
        :param power:
    
        sometimes you can get a more precise result from a matrix multiplication 
        by changing the order that matrixes are multiplied

        >>> parts = A._transformParts(N) 
        >>> assert parts[0]*parts[1] == A.transform(N)
        """
        if power == 0:
            vectors=self.vectors
            varP=numpy.ones_like(self.var)
        else:
            varP=numpy.real_if_close(self.var**(power/2.0))
            vectors=self.vectors

        return numpy.multiply(vectors.H,varP),vectors


    def transform(self,power=1):
        """
        :param power:
            
        >>> assert A.transform() == A.transform(1)
            
        >>> assert A.cov == (A**2).transform()
        >>> assert A.cov == A.transform()*A.transform()
        >>> assert A.cov == A.transform()**2
        >>> assert A.cov == A.transform(2)
        
        >>> assert A.transform(N)== (A**N).transform()
        >>> assert A.transform(N)== A.transform()**N  
        >>> #it's hit and miss for complex numbers, but real is fine
        >>> assert (A**numpy.real(K1)).transform() == A.transform(numpy.real(K1)) 

        >>> assert (A*B.transform() + B*A.transform()).cov/2 == (A*B).cov

        >>> assert Matrix(numpy.trace(A.transform(0))) == A.shape[0] 
        """
        if not self.var.size:
            ndim=self.ndim
            shape=(ndim,ndim)
            return Matrix.zeros(shape)


        assert (
            (self.var>0).all() or 
            int(power) == power
        ),"negative number cannot be raised to a fractional power"

        parts = self._transformParts(power)
        return parts[0]*parts[1]

    def sign(self):
        return helpers.sign(self.var)

    ########## Utilities
    def stack(*mvns,**kwargs):
        """
        :param * mvns:
        :param ** kwargs:
        
        >>> AB=A.stack(B)
        >>> assert AB[:,:A.ndim]==A
        >>> assert AB[:,A.ndim:]==B
        
        Stack two Mvns together, equivalent to hstacking the means, and 
        diag-stacking the covariances
        
        yes it works but be careful. Don't use this for reconnecting 
        something you calculated from an Mvn, back to the same Mvn it was 
        calculated from, you'll loose all the cross corelations. 
        If you're trying to do that use a better matrix multiply, or Mvn.chain 
        
        see also Mvn.chain
        """
        #no 'square' is necessary here because the rotation matrixes are in 
        #entierly different dimensions
        return type(mvns[0])(
            #stack the means
            mean=numpy.hstack([mvn.mean for mvn in mvns]),
            #stack the vector diagonally
            vectors=helpers.diagstack([mvn.vectors for mvn in mvns]),
            var=numpy.concatenate([mvn.var for mvn in mvns]),
            **kwargs
        )
    
    def sample(self,shape = (1,)):
        """
        :param shape:
            
        take samples from the distribution

        the vectors are aligned to the last dimension of the returned array
        
        >>> N = 5
        >>> assert A.sample(N).shape == (N,A.ndim)

        >>> N = 5,6,7
        >>> assert A.sample(N).shape == N+(A.ndim,)
                                
        a large number of samples will have the same mean and cov as the 
        Mvn being sampled

        >>> pows= reversed(range(1,6))
        >>> mvns = [Mvn.fromData(A.sample(5**P)) for P in pows] 
        >>> divergence = [m.KLdiv(A) for m in mvns]
        >>> assert Matrix(divergence) == sorted(divergence)

        """
        try:
            shape=list(shape)
        except TypeError:
            shape=[shape]
            
        shape.append(self.rank)

        #todo:that number should accept a size-tuple - returning size+ndim
        #todo look at that complex-normal distributions
        units = numpy.random.randn(*shape)

        return numpy.inner(units,self.scaled.H)+numpy.squeeze(numpy.array(self.mean))

    def measure(self,actual):
        """
        :param actual:
            
        This method is to simulate a sensor. 
     
        It treats the Mvn as a the description of a sensor, 
        the mean is the sensor's bias, and the variance is the sensor's variance.

        The result is an Mvn, with the mean being a sample pulled from the sensor's 
        distribution based on the 'actual' value you're trying to measure.

        It repreresents 1 measurment, the uncertianty interval is where you would expect 
        to find the actual value base on that single measurment.

        simulate sensor fusion:

        >>> sensor1=A
        >>> sensor1.mean*=0
        >>>
        >>> sensor2=B
        >>> sensor2.mean*=0
        >>>
        >>> actual=numpy.arange(0,A.ndim)
        >>>
        >>> result = sensor1.measure(actual) & sensor2.measure(actual)

        see also: Mvn.chain
        """
        finite = self[numpy.isfinite(self.var),:]        

        sample=finite.sample(1)-self.mean

        return self+(sample+actual)

    def det(self):
        """
        returns the determinant of the covariance matrix. 
        this method is supplied because the determinant can be calculated 
        easily from the variances in the object
        
        >>> assert Matrix(A.det())== numpy.linalg.det(A.cov)

        >>> assert A.det() == (
        ...     0 if 
        ...     A.rank!=A.ndim else 
        ...     A.var.prod()
        ... )

        if you want the pseudo-det use self.pdet
        (ref: http://en.wikipedia.org/wiki/Pseudo-determinant)
        """
        shape=self.shape
        return (
            0 if 
            shape[0]!=shape[1] else 
            self.var.prod()
        )

    def pdet(self):
        """
        returns the psudodet of the covariance matrix
        
        >>> assert A.pdet() == A.var.prod()
        """
        return self.var.prod()
        
    def trace(self):
        """
        returns the trace of the covariance matrix.
        this method is supplied because the trace can be calculated 
        easily from the variances in the object
        
        >>> assert Matrix(A.trace()) == numpy.trace(A.cov)
        >>> assert Matrix(A.trace()) == A.var.sum()

        >>> assert Matrix((A+B).trace()) == A.trace()+B.trace()
        
        >>> assert Matrix((A*B).trace()) == (B*A).trace()
        >>> assert Matrix((A.cov*B.cov).trace()) == (A*B).trace() 
        >>> assert Matrix((A*B.transform()).trace()) == (B*A.transform()).trace() 
        """
        return self.var.sum()
    
    def width(self):
        """
        return the standard deviations of the mvn, along each coordinate-axis.
        (not eigen-axes).
        
        >>> assert Matrix([A[:,n].var[0] for n in range(A.ndim)]) == A.width()**2

        >>> assert Matrix(A.corr.diagonal()) == Matrix.ones

        >>> norm = A/A.width()
        >>> assert norm.corr == norm.cov
        >>> assert Matrix([norm[:,n].var[0] for n in range(norm.ndim)]) == Matrix.ones

        This is very different from 

        >>> assert Matrix((A**0).var) == Matrix.ones

        because doing it with power scales along the eigenvectrs, this scales along the axes
        """
        S=numpy.array(self.scaled)
        return (S.conj()*S).sum(0)**(0.5)


    def chain(self,sensor=None,transform=None):
        """
        given a distribution of actual values and an Mvn to act as a sensor 
        this method returns the joint distribution of actual and measured values

        :param self:       is the value we're  taking a measurment of
        :param transform:  specifies the transform from the actual value to 
                           the sensor output defalults to Matrix.eye
        :param sensor:     specifies the sensor's bias and noise
                           defaults to Mvn.zeros

        so if you supply neither of the optional arguments you get:
        >>> assert A.chain()==A*numpy.hstack([E,E]) 
        
        if you have a perfect sensor (or no sensor) chain is just a different 
        way of doing a matrix multiply when you want to add new dimensions 
        to your data
        >>> assert ( 
        ...     A*numpy.hstack([E,M])== 
        ...     A.chain(transform=M)
        ... )

        when including a sensor, noise is added to those new dimensions

        >>> assert A.chain(B) == mooreChain(A,B)
        >>> assert A.chain(B*M,M) == mooreChain(A,B*M,M)

        some of te connections are more obvious when you look at it in terms of a block of data

        >>> dataA=A.sample(100)
        >>> a=Mvn.fromData(dataA)
        >>> assert a.chain()==Mvn.fromData(numpy.hstack([dataA,dataA]))        
        >>> assert a.chain(transform=M) == Mvn.fromData(dataA*numpy.hstack([E,M]))
        >>> assert a.chain(transform=M) == Mvn.fromData(numpy.hstack([dataA,dataA*M]))
        
        >>> assert a.chain(B*M,M) == a.chain(transform=M)+Mvn.stack(Mvn.zeros(a.ndim),B*M)

        see also : mvn.measure

        reference: andrew moore/data mining/gaussians
        """
        twice = type(self)(
            mean=numpy.hstack([self.mean,self.mean]),
            vectors=numpy.hstack([self.vectors,self.vectors]),
            var=self.var,
        )

        #create the base sensor output (before add is sensor noise)
        if transform is None:
            transform=Matrix.eye(self.ndim)
            perfect=twice
        else:
            Transform=helpers.diagstack([Matrix.eye(self.ndim),transform])
            perfect=twice*Transform
        

        #define the sensor noise
        sensor = sensor if sensor is not None else type(self).zeros(transform.shape[1])
        #add some leading seros so it fits
        sensor=type(self).zeros(self.ndim).stack(sensor)

        return perfect+sensor

    @decorate.MultiMethod
    def mah2(self,locations=None,mean=None):
        """
        :param locations:
        :param mean:
            
        Return the square of the mahalabois distance from the Mvn to each location.
        the vectors should be along the last dimension of the array.

        In this case the mah2 is the vector's length**2
        
            >>> E = Mvn.eye(A.ndim)
            >>> N = 50
            >>> S = numpy.random.randn(N,A.ndim)
            >>> assert Matrix(E.mah2(S)) == (S**2).sum(-1)

        This is Invariant to linear transforms
        
            >>> S=Matrix(A.sample(N))
            >>> T=Matrix.randn((A.ndim,A.ndim))
            >>> D1 = A.mah2(S)
            >>> D2 = (A*T).mah2(S*T)
            >>> assert Matrix(D1)==D2
        
        The expected distance squared of a sample from it's parent, is the 
        number of dimensions
        
            >>> A.mah2(A).mean = A.ndim
            
       for Mvns it currenty just returns an mvn (mean & variance), 
       but that should be a non-central chi**2 distribution

            >>> #warning: this works, but there is probably a better way.
            >>> N=1000
            >>> Z=3
            >>> deltas = Mvn.fromData(A.mah2(A.sample(N)) - A.ndim)
            >>> deltas.var/=N
            >>> assert deltas.mah2(0) < (Z**2)

        negative variances result in negative mah2's.

            >>> locations = B.sample([5,5])
            >>> assert -Matrix(A.mah2(locations)) == (~A).mah2(locations)
        """
        if callable(locations):
            locations = locations(self.mean.shape)

        locations = numpy.asarray(locations)
        
        if self.ndim == 1 and locations.ndim == 1 and locations.size != 1:
            locations = locations[:,None]

        if mean is None:
             mean = self.mean.squeeze()
        else:
             mean = numpy.array(mean).squeeze()            
            
        deltas=numpy.array(locations)-mean
        
        parts = self._transformParts(-2)        
        
        scaled = numpy.inner(deltas,parts[1])
        scaled = numpy.inner(scaled,parts[0])
        scaled=numpy.array(scaled)
        
#isn't there an easy way to write this in tensor notation?
        return (scaled*deltas).sum(axis=locations.ndim-1)
    
    @mah2.register(Mvn,type(None))
    def _mah2Self(self,locations,mean = None):
        """
        :param mean:                
        :param locations: no locations given, so it returns the distribution of 
                          the length of the self random vector
                          :type locations:`type(None)`
        
        """
        return scipy.stats.chi2(self.shape[0])
        
    @mah2.register(Mvn,Mvn)
    def _mah2Mvn(self,locations,mean = None):
        """        
        :param locations:
        :param mean:
        
        """    
        delta = (self + [-1]*locations)
#should be a generalized/non-central chi2
        return (delta/self).quad()
    
    def mah(self,locations=None,mean = None):
        """
        return the mahalabois distance from the mvn to each location
        
        .. plot:: ../examples/mah.py main

        """
        if locations is None:
            return scipy.stats.chi(self.shape[0])
        else:
            return self.mah2(locations)**0.5
        
    
    @decorate.MultiMethod
    def dist2(self,locations=None,mean = None):
        """
        :param locations:
        :param mean:
        """
        if callable(locations):
            locations = locations(self.mean.shape)

        locations = numpy.asarray(locations)
        
        if self.ndim == 1 and locations.ndim == 1 and locations.size != 1:
            locations = locations[:,None]
            print locations.shape

        
        if mean is None:
             mean = self.mean.squeeze()
        else:
             mean = numpy.array(mean).squeeze()            
            
        deltas=numpy.array(locations)-mean
        
#isn't there an easy way to write this in tensor notation?
        return (deltas*deltas).sum(axis=locations.ndim-1)


    @dist2.register(Mvn,type(None))
    def _dist2Self(self,locations,mean = None):
        """
        :param mean:        
        :param locations: no locations given, so it returns the distribution of the length of the self
            :type locations:`type(None)`
        
        
        """
        centered = self-self.mean
        return centered.quad()
        
    @dist2.register(Mvn,Mvn)
    def _dist2Mvn(self,locations,mean = None):
        """        
        :param locations:
        :param mean:
            
        """
#todo: find an implementation of generalized/noncentral chi2 distribution
        return (self + [-1]*locations).quad()
    
    def dist(self,locations=None,mean=None):
        """        
        :param locations:
        :param mean:
            
        returns the mahalabois distance to each location
        """
#todo: find an implementation of generalized/noncentral chi distribiution        
        return dist2(self,locations,mean)**0.5
        
    ############## indexing
    
    def given(self,dims,value=None):
        """
        :param dims:
        :params value:
            
        return an mvn representing the conditional probability distribution, 
        given the values, on the given dims

        also used for __setitem__
        
        equivalent to: andrew moore/data mining/gussians/page 22
        (except that my __and__ handles infinities)
        
        basic usage fixes the indexed component of the mean to the given value 
        with zero variance in that dimension.
        
        >>> a = A.given(dims=0,value=1)
        >>> assert a.mean[:,0]==1
        >>> assert a.vectors[:,0]==numpy.zeros

        Slices work
        
        >>> a = A.copy() #copy so that the inplace modification doesn't break other examples
        >>> a[:,1:] = 0
        >>> assert a.rank <= 1

        The value you're setting it to is irrelevant if you are only interested 
        in the variance:
        >>> assert A.given(dims=0,value=0).cov == A.given(dims=0,value=1000).cov

        This equivalent to doing an __and__ with an mvn of the apropriate shape
        zero var on the indexed dimensions, infinite vars on the others
        
        >>> L1=Mvn(mean=[0,0],vectors=[[1,1],[1,-1]], var=[numpy.inf,0.5])
        >>> L2=Mvn(mean=[1,0],vectors=[0,1],var=numpy.inf) 
        >>> assert L1.given(dims=0,value=1) == L1&L2
        >>> assert (L1&L2).mean==[1,1]
        >>> assert (L1&L2).cov==[[0,0],[0,2]]
        
        The above examples are with scalars but vectors work with apropriate 
        indexes
        
        because this is just an interface to __and__ the logical extension is:
        
        >>> Y=Mvn(mean=[0,1],vectors=Matrix.eye, var=[numpy.inf,1])
        >>> X=Mvn(mean=[1,0],vectors=Matrix.eye,var=[1,numpy.inf])
        >>> x=Mvn(mean=1,var=1)
        >>> assert Y.given(dims=0,value=x) == X&Y
        
        __setitem__ uses this for an inplace version
        
        >>> a=A.copy()
        >>> a[:,0]=1
        >>> assert a==A.given(dims=0,value=1)

        and remember that results get flattened by the slicing
        >>> assert A.rank > A.given(0 ,1).rank

        since it can accept values with a variance: 
        >>> x= Mvn(var = numpy.inf)
        >>> assert A.given(0,x) == A

        If you don't set a value something interesting happens
        >>> #!! this is one step from a reverse chain ?
        >>> dims = 0        
        >>> a = A.given(dims)
        >>> assert a == A.given(dims,~A[:,dims]) 
        >>> assert a  & A.marginal(dims) == A
        >>> assert A.given(0).given(0,0) == A.given(0,0)

        """
        #convert the inputs
        fixed=binindex(dims,self.ndim)
        N = fixed.sum()
        free = ~fixed

        #create the mean, for the new object,and set the values of interest
        if value is None:
            #value=type(self).zeros(n=N,mean = Matrix.nans)
            value=~self[:,dims] 
        else:                
            value=type(self).fromData(value)

    
        mean=Matrix.zeros([1,self.ndim])
        mean[:,fixed]=value.mean
        
        #create empty vectors for the new object
        vectors=numpy.zeros([
            value.shape[0]+(self.ndim-value.ndim),
            self.ndim,
        ])
        vectors[0:value.shape[0],fixed]=value.vectors
        vectors[value.shape[0]:,free]=numpy.eye(self.ndim-N)
        
        #create the variance for the new object
        var=numpy.zeros(vectors.shape[0])
        var[0:value.shape[0]]=value.var
        var[value.shape[0]:]=numpy.Inf

        #blend with the self
        return self & type(self)(
            var=var,
            mean=mean,
            vectors=vectors,
        ) 

        
    def __setitem__(self,index,value):
        """
        :param index:
        :params value:

        self[PCA,dims]=value
        
        This is an opertor interface to self.given 
        """
        PCA,dims = index
        PCA = binindex(PCA,self.ndim)

        keep = self[ PCA,:]
        drop = self[~PCA,:]
        keep = keep.given(dims,value)
        
        drop.mean[...] = 0
        
        self.copy(keep+drop)

    def marginal(self,index):
        """
        :param index:
            
        like __getitem__, but the result has the same dimensions as the self. 

        >>> assert A.marginal(slice(None)) == A

        >>> import operator
        >>> M = [A.marginal(n) for n in range(A.ndim)]
        >>> assert reduce(operator.and_,M) == A.diag()
        """
        index = ~binindex(index,self.ndim)
        N = index.sum()
        vectors = Matrix.zeros([N,self.ndim])
        vectors[range(N),index]=1
        return self+type(self)(
            var = Matrix.infs,
            vectors = vectors,
        )
        

    def __getitem__(self,index):
        """
        :param index:
        
        self[:,index]
        
        return the marginal distribution, over the indexed dimensions.
        
        self[:N,:]
        
        return a distribution using only the first N principal components.
        
        >>> data = Matrix.randn([1000,2])*(Matrix.eye(2)+Matrix.randn([2,2]))+Matrix.randn([1,2])
        >>> XY = Mvn.fromData(data)
        >>> X = Mvn.fromData(data[:,0])
        >>> Y = Mvn.fromData(data[:,1])
        >>> assert X == XY[:,0]
        >>> assert Y == XY[:,1]
        """
        if isinstance(index,tuple):
            PCA,DIMS = index
        else:
            PCA = index
            DIMS = slice(None)
         
        DIMS = numpy.asarray(DIMS) if hasattr(DIMS,'__iter__') else DIMS
        PCA  = numpy.asarray( PCA) if hasattr(PCA ,'__iter__') else PCA
        
        return type(self)(
            mean=self.mean[:,DIMS],
            vectors=self.vectors[PCA,DIMS],
            var=self.var[PCA],
        )

    ############ Math

    def __eq__(self,other):
        """
        :param other:
            
        self == other

        mostly it does what you would expect

        >>> assert A==A.copy()
        >>> assert A is not A.copy()
        >>> assert A.mean is A.copy().mean
        >>> assert A.mean is not A.copy(deep = True).mean
        >>> assert A != B

        You'll get an Error if the mvns have different numbers of dimensions. 
        
        Infinite, and zero variances are handled correctly.
        
        Note that the component of the mean along a direction with infinite variance is ignored:

            >>> assert (
            ...     Mvn(mean=[1,0,0], vectors=[1,0,0], var=numpy.inf)==
            ...     Mvn(mean=[0,0,0], vectors=[1,0,0], var=numpy.inf)
            ... )

        __ne__ is handled by the Automath class

            >>> assert A != B

        when the comparison is done with a function, the function is called 
        with the size of the self as the only argument:
            
            >>> n=abs(N)
            >>> assert Mvn.zeros(n) == Mvn.zeros
            >>> assert Mvn.eye(n) == Mvn.eye
            >>> assert Mvn.infs(n) == Mvn.infs
        """
        if callable(other):
            other = other(self.ndim)

        other=type(self).fromData(other)
        
        #check the number of dimensions of the space
        assert  self.ndim == other.ndim,"""
            if the objects have different numbers of dimensions, 
            you're doing something wrong
            """

        self=self.squeeze()
        other=other.squeeze()

        Sshape=self.shape
        Oshape=other.shape

        #check the number of flat dimensions in each object
        if Sshape[0]-Sshape[1] != Oshape[0] - Oshape[1]:
            return False

        
        Sfinite=numpy.isfinite(self.var)
        Ofinite=numpy.isfinite(other.var)

        if Sfinite.sum() != Ofinite.sum():
            return False

        if Sfinite.all():
            return self.mean==other.mean and self.cov == other.cov
    
        #remove the infinite directions from the means
        SIvectors=self.vectors[~Sfinite]
        Smean=self.mean - self.mean*SIvectors.H*SIvectors 

        OIvectors=other.vectors[~Ofinite]
        Omean=other.mean - other.mean*OIvectors.H*OIvectors

        #compare what's left of the means   
        if Smean != Omean:
            return False

        SFvectors=self.vectors[Sfinite]
        SFvar=self.var[Sfinite]

        OFvectors = other.vectors[Ofinite]

        cov=lambda vectors,var: numpy.multiply(vectors.H,var)*vectors

        #compare the finite and infinite covariances 
        return (
            cov(SFvectors,SFvar) == cov(OFvectors,SFvar) and
            SIvectors.H*SIvectors == OIvectors.H*OIvectors
        )

    def __gt__(self,lower):
        """
        :param lower:
            
        >>> assert Matrix(A > Matrix.infs(A.ndim)) == 0
        >>> assert Matrix(A >-Matrix.infs(A.ndim)) == 1
        
        >>> AV = A*A.vectors.H
        >>> assert Matrix(AV>AV.mean) == 2**-AV.ndim 

        see :py:meth:`mvn.Mvn.inbox`
        """
        return self.inBox(
            lower,
            Matrix.infs(self.mean.size)
        )
        
    def __ge__(self,lower):
        """
        :param lower:
            
        see :py:meth:`mvn.Mvn.gt`
        see :py:meth:`mvn.Mvn.inbox`
        """
        return self>lower

    def __lt__(self,upper):
        """
        :param upper:
            
        >>> assert Matrix(A < Matrix.infs(A.ndim)) == 1
        >>> assert Matrix(A <-Matrix.infs(A.ndim)) == 0
        
        >>> AV = A*A.vectors.H
        >>> assert Matrix(AV<AV.mean) == 2**-AV.ndim 

        see :py:meth:`mvn.Mvn.inbox`
        """
        return self.inBox(
            -Matrix.infs(self.mean.size),
            upper,
        )

    def __le__(self,lower):
        """
        :param lower: 
            
        see :py:meth:`mvn.Mvn.lt`
        see :py:meth:`mvn.Mvn.inbox`
        """
        return self>lower

    def inBox(self,lower,upper,**kwargs):
        """
        :param lower:
        :param upper:
            
        returns the probability that all components of a sample are between the 
        lower and upper limits 

#todo: fix

        >>> N = 100
        >>> data = A.sample(N)
        >>> limits = A.sample(2)
        >>> upper = limits.max(0)
        >>> lower = limits.min(0)
        >>> print Mvn.mean(((data<upper) & (data>lower)).all(1))
        >>> print A.inBox(lower,upper)
        >>> assert A.inBox(lower,upper) == Mvn.mean(((data<upper) & (data>lower)).all(1))

#todo: this could be expanded to return a gaussian mixture, 
              with one (Mvn) component instead of just a  weight...
        """
#todo: vectorize?
        lower=lower-self.mean
        upper=upper-self.mean

#todo: multimethod?
        if isinstance(lower,Mvn):
            l=lower.mean
            lower.mean=Matrix.zeros
            self = self+lower
            lower=l
            
        if isinstance(upper,Mvn):
            u=upper.mean
            upper.mean=Matrix.zeros
            self = self+upper
            upper=u

        lower=numpy.array(lower)
        upper=numpy.array(upper)
        
        if (lower == upper).any():
            return 0.0
            
        upper = upper.squeeze()
        lower = lower.squeeze()
        
        upInfs = numpy.isinf(upper)
        lowInfs = numpy.isinf(lower)        
        bothInfs = (upInfs & lowInfs)
        
        if bothInfs.any():
                
            #But any time the inf in upper is negative, the sign 
            #is inverted, so count the inversions
            infFlips =(-1.0)**(
                numpy.sign(upper[bothInfs])< 
                numpy.sign(lower[bothInfs])
            ).sum()
            
            #if we've made it here, any slots that are both inf 
            #have different signs, so it's like we're integrating 
            #the marginal
            self = self[:,~bothInfs]
            upper = upper[~bothInfs]        
            lower = lower[~bothInfs]
            
            if not self:
                return infFlips
                
        else:
            infFlips = 1.0 
        
        
        if self.ndim == 1:
            gaus1D=scipy.stats.norm(0,self.var**0.5)
            return gaus1D.cdf(upper)-gaus1D.cdf(lower)

        Iwidth=self.width()**-1
        
        self = self*Iwidth
        lower = lower*Iwidth
        upper = upper*Iwidth

        return infFlips*mvncdf.mvstdnormcdf(lower,upper,self.corr,**kwargs)
        
    def bBox(self,nstd=2):
        """
        :param nstd:
            
        return a 2xndim Matrix where the frst row is mean-n*std, 
        and the second row is mean+n*std
        >>> nstd = 2
        >>> assert A.bBox(nstd) == Matrix.stack([
        ...     [A.mean-nstd*A.width()],
        ...     [A.mean+nstd*A.width()]
        ... ])
        """
        return Matrix.stack([
            [self.mean-nstd*self.width()],
            [self.mean+nstd*self.width()],
        ])
        
    def __abs__(self):
        """
        abs(self)

        sets all the variances to positive
        >>> assert (A.var>=0).all()
        >>> assert abs(A) == abs(~A)
        
        but does not touch the mean
        >>> assert Matrix((~A).mean) == Matrix(abs(~A).mean)
        """
        result=self.copy()
        result.var=abs(self.var)
        return result
    
    def __invert__(self):
        """
        ~self

        todo: signed inf's
        
        implementation:     
   
            >>> IA=A.copy(deep=True)
            >>> IA.var*=-1
            >>> assert IA == ~A
            
        so:
            >>> assert (~A).mean == A.mean
            >>> assert Matrix((~A).var) == (-A).var 
            >>> assert Matrix((~A).var) == -(A.var)

            >>> assert (~A).vectors==A.vectors

            >>> assert (~A).cov == (-A).cov 
            >>> assert (~A).cov == -(A.cov)

            >>> assert ~~A==A

            >>> assert ~(~A&~B) == A & B 
 
        something and not itself provides zero precision; infinite variance
        except if the mvn is flat; that flattness is preserved 
        (because zeros get squeezed, positive and negative zeros are indistinguishable)

        >>> assert (A & ~A) == Mvn(mean=A.mean, vectors=A.vectors, var=Matrix.infs)
        >>> if not A.flat:
        ...     assert (A & ~A) == Mvn(mean=numpy.zeros(A.ndim))**-1


        infinite variances provide no information, having a no effect when blended
        >>> if not B.flat:
        ...     assert A == A & (B & ~B)
        
        if the mvn is flat, things are a little different:
            like this you're taking a slice of A in the plane of B
            >>> assert  A &(B & ~B) == A & Mvn(mean=B.mean, vectors=B.vectors, var=Matrix.infs)
   
            but watch out:
            >>> assert (A&~B) & B == (A&B) & ~B
            >>> if not (A.flat or B.flat):
            ...    assert (A&B) & ~B == A & (B&~B)

            todo: investigate why this doesn't match, 
                I don't have a sense-physique for negative variances,
                so I don't have a good explaination right now, it's probably related 
                to the fact that (B&~B) removes all information about he variance of B  
                but some seems to be preserved in the 2 stage operation

        so blending something with it's inverse is equivalend to replacing all 
        it's non-zero variances with inf's

        >>> assert not numpy.isfinite((A & ~A).var).any()

        >>> P=A.copy()
        >>> P.var=P.var/0.0
        >>> assert P==(A & ~A)       

        the automath logic extensions are actually useless to Mvn because:
            >>> assert (~A & ~B) == ~(A & B)

            so 'or' would become a copy of 'and' and 'xor' would become a blank equavalent to the (A & ~A) above

            maybe the A|B = A+B - A&B  version will be good for something; I'll put them in for now
        """
        result=self.copy()
        result.var=-(self.var)
        return result
    
    def __or__(self,other):
        """
        :param other:
            
        self | other
        >>> assert  (A | B) == (A+B) - (A&B)
        """
        return (self+other)-(self&other)

    def __xor__(self,other):
        """
        :param other:
            
        I don't  know what this means yet
        """
        return self+other-2*(self&other)

    def __and__(self,other):
        """
        :param other:
            
        self & other
        
        This is awsome.
        
        it is the blend step from kalman filtering
        
        optimally blend together two mvns, this is done with the 
        'and' operator because the elipses look like ven-diagrams
        
        Just choosing an apropriate inversion operator (1/A) allows us to 
        define kalman blending as a standard 'paralell' operation, like with 
        resistors. operator overloading takes care of the rest.
        
        The inversion automatically leads to power, multiply, and divide  
        
        >>> assert A & B == B & A 

        >>> if not (A.flat or B.flat): 
        ...     assert A & B == 1/(1/A+1/B)

        >>> if not (A.flat or B.flat or C.flat):
        ...     import operator
        ...     abc=numpy.random.permutation([A,B,C])
        ...     assert A & B & C == helpers.paralell(*abc)
        ...     assert A & B & C == reduce(operator.and_ ,abc)
        ...     assert (A & B) & C == A & (B & C)

        >>> if not A.flat:
        ...     assert A &-A == Mvn(mean=numpy.zeros(ndim))**-1
        ...     assert A &~A == Mvn(mean=numpy.zeros(ndim))**-1


        >>> assert (A & A).cov == A.cov/2
        >>> assert (A & A).mean == A.mean
                
        The proof that this is identical to the wikipedia definition of blend 
        is a little too involved to write here. Just try it (and see the "wiki"
        function)
        
        >>> if not (A.flat or B.flat):
        ...     assert A & B == wiki(A,B)

        this algorithm is also, at the same time, solving linear equations
        where the zero variances correspond to a plane's null vectors 

        >>> L1=Mvn(mean=[1,0],vectors=[0,1],var=numpy.inf)
        >>> L2=Mvn(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[1,1]
        >>> assert (L1&L2).var.size==0

        >>> L1=Mvn(mean=[1,0],vectors=[1,1],var=numpy.inf)
        >>> L2=Mvn(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[2,1]
        >>> assert (L1&L2).var.size==0
        
        >>> L1=Mvn(mean=[0,0],vectors=Matrix.eye, var=[1,1])
        >>> L2=Mvn(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[0,1]
        >>> assert (L1&L2).var==1
        >>> assert (L1&L2).vectors==[1,0]
        
    """
        #check if they both fill the space
        if not (self.flat or other.flat):
            #then this is a standard paralell operation
            result=(self**(-1)+other**(-1))**(-1) 
        else:
            Dmean = Plane.__and__(self,other).mean

            #do the blending, while compensating for the mean of the working plane
            result=((self-Dmean)**-1+(other-Dmean)**-1)**-1+Dmean

        return result

    def __pow__(self,power):
        """
        :param power:
            
        self**power

        >>> #the transform version doesn't work for flat objects if the transform power is less than 0
        >>> k = numpy.real(K1)
        >>> if not A.flat or k>0:
        ...     assert A**k == A*A.transform(k-1) + Mvn(mean=A.mean-A.mean*A.transform(0)) 

        This definition was developed to turn kalman blending into a standard 
        resistor-style 'paralell' operation
        
        The main idea is that only the variances get powered.
        (which is normal for diagonalizable matrixes), stretching the sheet at 
        at an aprpriate rate along each (perpendicular) eigenvector
        
        Because powers on the variance are easy, this is not restricted to 
        integer powers
        
        But the mean is also affected by the stretching. It's as if the usual 
        value of the mean is a "zero power mean" transformed by whatever is 
        the current value of the self.scaled matrix and if you change that 
        matrix the mean changes with it..
        
        Most things you expect to work just work.

            >>> assert A==A**1
            >>> assert -A == (-A)**1

            >>> #this doesn't work for flat objects because information about 
            >>> #the mean is lost after the first inversion, because of the infinite variance.
            >>> if not A.flat:
            ...     assert A == (A**-1)**-1
            >>> assert A.mean*A.transform(0) == ((A**-1)**-1).mean
    
            >>> assert A**0*A==A
            >>> assert A*A**0==A
            >>> if not A.flat:
            ...     assert A**0 == A**(-1)*A
            ...     assert A**0 == A*A**(-1)
            ...     assert A**0 == A/A 
            ...     assert A/A**-1 == A**2
            
            those only work if the k's are real      
            >>> k1=numpy.real(K1)
            >>> k2=numpy.real(K2)
            >>> if ((not A.flat) or (k1>=0 and k1>=0)):
            ...     assert (A**k1)*(A**k2)==A**(k1+k2) 
            ...     assert A**k1/A**k2==A**(k1-k2) 
            
        Zero power has some interesting properties: 
            
            The resulting ellipse is always a unit sphere, 
            the mean is wherever it gets stretched to while we 
            transform the ellipse to a sphere
              
            >>> assert Matrix((A**0).var) == numpy.ones
            >>> assert (A**0).mean == A.mean*(A**-1).transform() if not A.flat else True

            if there are missing dimensions the transform is irreversable so this stops working 
            >>> assert (A**0).mean == A.mean*A.transform(-1) or A.flat
            
        derivation of multiplication from this is messy.just remember that 
        all Mvns on the right, in a multiply, can just be converted to matrix:
            
            >>> assert (A*B).cov == (A*B.transform()+B*A.transform()).cov/2

            >>> assert (A*B).mean == (A*B.transform()+B*A.transform()).mean/2 or (A.flat or B.flat)

            >>> assert M.H*B==M.H*B.transform()
            >>> assert A**2==A*A
            >>> if not A.flat:
            ...     #information is lost if the object is flat.
            ...     assert A**2==A*A.transform()
        """
        assert numpy.isreal(power)
        power = numpy.real(power)

        transform = self.transform(power-1)

        if numpy.real(power)<0: 
            self=self.inflate()
        
        V=self.vectors            
        dmean=self.mean-self.mean*V.H*V        
        
        return type(self)(
            mean=self.mean*transform+dmean,
            vectors=self.vectors,
            var=self.var**power,
            square=False,
        )

    @decorate.prepare(lambda self,other:(self,type(self).format(other)))
    @decorate.MultiMethod
    def __mul__(self,other):        
        """
        :param other:
            
        self*other
        
        coercion notes:
            All non Mvn imputs will be converted to numpy arrays, then 
            treated as constants if zero dimensional, or matrixes otherwise.

            the resulting types for all currently supported multipications are listed below            
            
            >>> assert isinstance(A*B,Mvn)
            >>> assert isinstance(A*M,Mvn)
            >>> assert isinstance(M.T*A,Matrix) 
            >>> assert isinstance(A*K1,Mvn)
            >>> assert isinstance(K1*A,Mvn)

            This can be explained as: 

                When multiplying by a constant the result is always an Mvn.
                
                When multiplying a mix of Mvns an Matrixes the result has the 
                same type as the leftmost operand
            
       Whenever an mvn is found on the right of a Matrix or Mvn it is replaced by a 
            self.transform() matrix and the multiplication is re-called.
            
        general properties:
            
            Scalar multiplication fits with addition so:
                
                >>> assert A+A == 2*A
                >>> assert (2*A).mean==2*A.mean
                >>> assert (2*A.cov) == 2*A.cov
            
            This is different from multiplication by a scale matrix which gives
            
                >>> assert (A*(K1*E)).mean == K1*A.mean
                >>> assert (A*(K1*E)).cov == A.cov*abs(K1)**2

            1d arrays in a multipliction are interpreted as the diagonal in 
            a matrix multiply
            
                >>> assert (A*(K1*E)).mean == (A*[K1]).mean
                >>> assert (A*(K1*E)).cov == (A*[K1]).cov

            constants still commute:          

                >>> assert K1*A*M == A*K1*M 
                >>> assert K1*A*M == A*M*K1

            constants are still asociative

                >>> assert (K1*A)*K2 == K1*(A*K2)

            so are matrixes if the Mvn is not in the middle, because it's all matrix multiply.

                >>> assert (A*M)*M2.H == A*(M*M2.H)
                >>> assert (M*M2.H)*A == M*(M2.H*A)

            if you mix mvns with matrixes, it's two different types of multiplication, and 
            so is not asociative
                
            the reason that those don't work boils down to:            

                >>> am=A.transform()*M
                >>> ma=(A*M).transform()
                >>> assert am.shape != ma.shape or ma != am

            if you work it out you'll find that the problem is unavoidable given:

                >>> assert (A*M).cov == M.H*A.cov*M
                >>> assert (A**2).transform() == A.cov

            multiplication is distributive for constants only.

                >>> assert A*(K1+K2)==A*K1+A*K2

                >>> assert A*(M+M2)!=A*M+A*M2
                
                >>> assert A*(B+C)!=A*B+A*C
             
                The reason is more clear if you consider the following:

                >>> assert A*(E+E) != A*E+A*E

                because the left side will do a matrix multiplication by 2, 
                and the right will do a scalar multiplication by 2, the means will match but the cov's will not 
                
                The pure mvn case fails for slightly different reasons:
                
                    >>> assert A*(B**0+B**0) == A*(2*B**0)   #here the mean is stretched to sqrt(2) times 
                    >>> assert (2*B**0).transform() == sqrt(2)*(B**0).transform()    
        
                    >>> assert (A*B**0 + A*B**0).cov == (2*A*B**0).cov 
                    >>> assert (A*B**0 + A*B**0).mean != (2*A*B**0).mean
            
        given __mul__ and __pow__ it would be immoral to not overload divide as well, 
        The Automath class takes care of these details
            
            A/?
            
                >>> m=M*M2.H
                >>> assert A/m == A*(m**(-1))            
                >>> assert A/B == A*(B**(-1))
                >>> assert A/K1 == A*(K1**(-1))
        
            ?/A: see __rmul__ and __pow__
            
                >>> assert K1/A == K1*(A**(-1))
                >>> assert M.H/A==M.H*(A**(-1))

                >>> assert Matrix((A**0.0).trace()) == A.shape[0]
        """
        return NotImplemented

    @decorate.prepare(lambda self,other:(self,type(self).format(other)))
    @decorate.MultiMethod    
    def __rmul__(self,other):
        """
        :param other:
            
        other*self
        
        multiplication order doesn't matter for constants
        
            >>> assert K1*A == A*K1
        
            but it matters a lot for Matrix/Mvn multiplication
        
            >>> assert isinstance(A*M,Mvn)
            >>> assert isinstance(M.H*A,Matrix)
        
        be careful with right multiplying:
            Because power must fit with multiplication
        
            it was designed to satisfy
            >>> assert A*A==A**2
        
            The most obvious way to treat right multiplication by a matrix is 
            to do exactly the same thing we're dong in Mvn*Mvn, which is 
            convert the right Mvn to the square root of its covariance matrix
            and continue normally,this yields a matrix, not an Mvn.
            
            this conversion is not applied when multiplied by a constant.
        
        martix*Mvn
            >>> assert M.H*A==M.H*A.transform()

        Mvn*constant==constant*Mvn
            >>> assert A*K1 == K1*A
        
        A/?
        
        see __mul__ and __pow__
        it would be immoral to overload power and multiply but not divide 
            
            >>> m=M*M2.H
            >>> assert A/B == A*(B**(-1))
            >>> assert A/m == A*(m**(-1))
            >>> assert A/K1 == A*(K1**(-1))

        ?/A
        
        see __rmul__ and __pow__
            >>> assert K1/A == K1*(A**(-1))
            >>> assert M.H/A==M.H*(A**(-1))
        """
        return NotImplemented

    
    @__mul__.register(Mvn)
    @__rmul__.register(Mvn)
    def _scalarMul(self,scalar):
        """
        :param scalar:
            
        self*scalar, scalar*self

        >>> assert A*K1 == K1*A

            Matrix multiplication and scalar multiplication behave differently 
            from eachother.  
            
            For this to be a properly defined vector space scalar 
            multiplication must match with addition, and addition here is 
            defined so it can be used in the kalman noise addition step so: 
            
            >>> assert (A+A)==(2*A)
            
            >>> assert (A+A).mean==(2*A).mean
            >>> assert (A+A).mean==2*A.mean
            
            >>> import itertools
            >>> assert sum(itertools.repeat(A,N-1),A) == A*(N) or N<=0

            after that the only things you're really guranteed here are:
            >>> assert (A*K1).mean==K1*A.mean
            >>> assert (A*K1).cov== (A.cov)*K1
                        
            if you don't like imaginary numbes be careful with negative constants because you 
            will end up with imaginary numbers in your ... stuf.            
            
            >>> assert B+(-A) == B+(-1)*A == B-A
            >>> assert (B-A)+A==B
            
            if you want to scale the distribution linearily with the mean
            then use matrix multiplication
        """
        assert numpy.isreal(scalar)
        scalar = numpy.real(scalar)

        return type(self)(
            mean= scalar*self.mean,
            var = scalar*self.var,
            vectors = self.vectors,
            square = not numpy.isreal(scalar),
        )

    @__mul__.register(Mvn,Matrix)
    def _matrixMul(self,matrix):
        """
        :param matrix:
        
        self*matrix
        
            matrix multiplication transforms the mean and ellipse of the 
            distribution. Defined this way to work with the kalman state 
            update step.

            >>> assert A*E==A
            >>> assert (-A)*E==-A 
            
            there is a shortcut for diagonal matrixes:
            >>> assert A*(E*K1)== A*[K1]
            >>> assert A*(numpy.diagflat(1/A.width())) == A/A.width()
            
            or with a more general transform
            >>> assert (A*M).cov==M.H*A.cov*M
            >>> assert (A*M).mean==A.mean*M

        """
        return type(self)(
            mean=self.mean*matrix,
            var=self.var,
            vectors=self.vectors*matrix,
        )

    @__rmul__.register(Mvn,Matrix)
    def _rmatrixMul(self,matrix):
        """
        :param matrix:
        """
        return matrix*self.transform()

    @__mul__.register(Mvn,Mvn)
    @__rmul__.register(Mvn,Mvn)
    def _mvnMul(self,mvn):
        """
        :param mvn:
            
        self*mvn

        multiplying two Mvns together is defined to fit with power
        
        >>> assert A*A==A**2
        >>> if not A.flat:
        ...     assert A*A==A*A.transform()
        >>> assert A*B == B*A

        Note that the result does not depend on the mean of the 
        second mvn(!) (really any mvn after the leftmost mvn or matrix)
        """
#revert_multiply -> should match transforming data.
        self0,self1 = self._transformParts()
        mvn0,mvn1 = mvn._transformParts()

        result = (self*mvn0*mvn1+mvn*self0*self1)
        
        result.mean += (
            self.mean-self.mean*mvn.transform(0)+
            mvn.mean-mvn.mean*self.transform(0)
        )

        return result/2

    @__mul__.register(Mvn,numpy.ndarray)
    @__rmul__.register(Mvn,numpy.ndarray)
    def __vectorMul__(self,vector):
        """
        :param vector:
            
        >>> assert A*range(A.ndim) == A*numpy.diagflat(range(A.ndim))
        >>> assert A+[-1]*A == A+A*(-1*E)
        """

        assert (vector.ndim == 1), 'vector multiply, only accepts 1d arrays' 
        assert (vector.size == 1 or  vector.size == self.ndim),'vector multiply, vector.size must match mvn.ndim'
        
        return type(self)(
            mean=numpy.multiply(self.mean,vector),
            vectors=numpy.multiply(self.vectors,vector),
            var=self.var,
        )
            

    def quad(self,matrix=None):
        #todo: noncentral Chi & Chi2 distribution gives the *real* distribution 
        #       of the length & length^2 this just has the right mean and
        #       variance
        """
        :param matrix:
            
        ref: http://en.wikipedia.org/wiki/Quadratic_form_(statistics)

        when used without a transform matrix this will get you the distribution 
        of the vector's magnitude**2.

        use this to dot an mvn with itself like (rand())**2 
        use iner if you want rand()*rand()

        If you're creative with the marix transform you can make lots of 
            interesting things happen

        todo: Chi & Chi2 distribution gives the *real* distribution of the length & length^2
                this has the right mean and variance so it's like a maximul entropy model
                (ref: http://en.wikipedia.org/wiki/Principle_of_maximum_entropy )
        """
        if matrix is not None:
            matrix=(matrix+matrix.H)/2

        transformed = self if matrix is None else self*matrix
        flattened   = (transformed*self.mean.H).inflate()

        return type(self)(
            mean=flattened.mean+numpy.trace(self.cov if matrix is None else matrix*self.cov) ,
            var=4.0*flattened.var+2.0*numpy.trace(transformed.cov*self.cov),
        )

    #todo: add a test case to show why quad and dot are different
    #todo: add a 'transposed' class so inner is just part of multiply

    @__mul__.register(Mvn,Mvn.T)
    def inner(self,other):
        """
        :param other:
            
        >>> assert A.inner(B) == B.inner(A)

        use this to dot product two mvns together, dot is like rand()*rand()
        be careful dot producting something with itself you you might actually 
        want rand()**2, use mvn.quad for that
        """        
        return type(self)(
            mean=self.mean*other.mean.H,
            var=(
                (self*other).trace() + 
                (other*self.mean.H).trace() + 
                (self*other.mean.H).trace()
            )
        )
    

    #todo: add a 'transposed' class so outer is just part of multiply
    @__mul__.register(Mvn.T,Mvn)
    def outer(self,other):
        """
        :param other:
            
        >>> assert A.outer(B).trace() == A.inner(B).mean
        >>> assert A.outer(B) == B.outer(A).T
        """
        return Matrix(numpy.outer(self.mean,other.mean))

    @decorate.MultiMethod
    def __add__(self,other):
        """
        :param other:
            
        self+other
        
        When using addition keep in mind that rand()+rand() is not like scaling 
        one random number by 2 (2*rand()), it adds together two random numbers.

        The add here is like rand()+rand()
        
        Addition is defined this way so it can be used directly in the kalman 
        noise addition step, it also makes these things a real vector space
        
        so if you want simple scale use matrix multiplication like:
            >>> assert A*[2] == A*(2*Matrix.eye(A.ndim))

        scalar multiplication however fits with addition:
            >>> assert A+A == A*2

            >>> assert (A+A).mean == (2*A).mean
            >>> assert (A+A).mean == 2*A.mean
        
            >>> assert (A+B).mean == A.mean+B.mean
            >>> assert (A+B).cov == A.cov+B.cov
        
        and watch out subtraction is the inverse of addition 
            >>> assert A-A == Mvn(mean=numpy.zeros_like(A.mean))
            >>> assert (A-B)+B == A
            >>> assert (A-B).mean == A.mean - B.mean
            >>> assert (A-B).cov== A.cov - B.cov
            
        if you want something that acts like rand()-rand() use an eye to scale:
            
            >>> assert (A+B*(-E)).mean == A.mean - B.mean
            >>> assert (A+B*(-E)).cov== A.cov + B.cov

        use arraymultiply as a shortcut for and diagonal matrix multiply
            >>> assert -1*A==-A
            >>> assert -A != A*(-E)
            >>> assert A*(-E) == [-1]*A

        __sub__ should also fit with __neg__, __add__, and scalar multiplication.
        
            >>> assert B+(-A) == B+(-1)*A == B-A
            >>> assert A-B == -(B-A)
            >>> assert A+(B-B)==A
            
        """
        #todo: fix the crash generated, for flat objects by: 1/A-1/A (inf-inf == nan)
        raise TypeError('Not implemented for these types')

    @__add__.register(Mvn)
    def _addDefault(self,other):
        """        
        :param other:
        """
        result = self.copy()
        result.mean = result.mean + other 
        return result

    @__add__.register(Mvn,Mvn)
    def _addMvn(self,other):
        """
        :param other:
            
        Implementation:
            >>> assert (A+B)==Mvn(
            ...     mean=A.mean+B.mean,
            ...     vectors=numpy.vstack([A.vectors,B.vectors]),
            ...     var = numpy.concatenate([A.var,B.var]),
            ... )
        """
        return type(self)(
            mean=self.mean+other.mean,
            vectors=numpy.vstack([self.vectors,other.vectors]),
            var = numpy.concatenate([self.var,other.var]),
        )


        
    def density(self,locations):
        """
        :param locations:
            
        self(locations)

        Returns the probability density in the specified locations, 
        The vectors should be aligned onto the last dimension
        That last dimension is squeezed out during the calculation

        >>> data = A.sample([5,5])
        >>> assert Matrix(A.density(data)) == numpy.exp(-A.entropy(data))

        >>> data = (A&B).sample([10,10])
        >>> a=A.density(data)
        >>> b=B.density(data)
        >>> ab = (A&B).density(data)
        >>> ratio = (a*b)/ab
        >>> assert Matrix(0) == ratio.var()
        """
        return numpy.exp(-self.entropy(locations)) 
    

    def entropy(self,data=None,base=None):
        """
        :param data:
        :param base:
            
        information required to encode A using a code based on B
        definition:

            A.entropy(B) -> sum(p(B)*log(p(A)))

        This returns the differential entropy.        
        The default base is stored in the class-attribute 'infoBase'

            >>> assert A.infoBase is A.__class__.infoBase
            >>> assert A.infoBase is numpy.e

        With vectors it is simply the log of the inverse of the probability of each sample

            >>> Sa=A.sample(100)   
            >>> assert Matrix(numpy.log(1/A.density(Sa))) == A.entropy(Sa)
            >>> Sb=B.sample(100)   
            >>> #sometimes this next one is just a bunch of infinities
            >>> assert Matrix(1/A.density(Sb)) == numpy.exp(A.entropy(Sb)) 

        With an Mvn it is the average encoding length per sample (it would be 
            nice if I could find the distribution instead of just the mean)

            >>> assert Matrix(A.entropy()) == A.entropy(A) 
            >>> assert Matrix(A.entropy()) == A.entropy(A.copy())

        Entropy is sensitive to linear transforms
            >>> m=Matrix.randn([A.ndim,A.ndim])
            >>> assert Matrix((A*m).entropy()) == A.entropy()+numpy.log(abs(m.det()))
           
        But the difference is proportional to the det of the transform so 
        rotations for example do nothing.
        
            >>> assert Matrix(A.entropy()) == (A*A.vectors.H).entropy()

        The joint entropy is less than the sum of the marginals

            >>> marginalEntropy = Matrix([
            ...     A[:,dim].entropy() 
            ...     for dim in range(A.ndim)
            ... ])
            >>> assert Matrix(A.diag().entropy()) == marginalEntropy.sum()
            >>> assert A.entropy() <= marginalEntropy.sum() 

        At the limit, a the mean mean entropy of a large number of samples from 
            a distribution will be the entropy of the distribution 

            warning: this works, but there is probably a better way.
            
            note: the variance of the mean of the sample is the variance 
                  of the sample divided by the sample size.

            >>> N=1000
            >>> Z=3
            >>>
            >>> a=A.sample(N)
            >>> deltas = Mvn.fromData(A.entropy(a)-A.entropy())
            >>> deltas.var/=N
            >>> assert deltas.mah2(0) < (Z**2)     
            >>>
            >>> b=B.sample(N)
            >>> deltas = Mvn.fromData(A.entropy(b)-A.entropy(B))
            >>> deltas.var/=N
            >>> assert deltas.mah2(0) < (Z**2)  

        http://en.wikipedia.org/wiki/Multivariate_normal_distribution
        """
        if base is None:
            base=self.infoBase

        if data is None:
            data = self
#todo: multimethod
        if isinstance(data,Mvn):
            baseE = (
                numpy.log(abs(data.pdet()))+
                data.rank*numpy.log(2*numpy.pi*numpy.e)
            )/2
            if data is not self:
                baseE+=self.KLdiv(data)
        else:
            baseE=(
                self.mah2(data)+
                self.rank*numpy.log(2*numpy.pi)+
                numpy.log(abs(self.pdet()))
            )/2

        return baseE/numpy.log(base)
        
        
    def KLdiv(self,other,base=None):
        """
        :param other:
        :param base:
            
        A.KLdiv(B) -> sum(p(B)*log(p(B)/p(A)))

        Return the KLdiv in the requested base.        
        The default base is stored in the class-attribute 'infoBase'

            >>> assert A.infoBase is A.__class__.infoBase
            >>> assert A.infoBase is numpy.e

        it measures the difference between two distributions:
        there is no difference between a distribution and itself        

            >>> assert Matrix(A.KLdiv(A)) == 0 
    
        The difference is always positive

            >>> assert A.KLdiv(B) > 0

        Invariant under linear transforms

            >>> m = Matrix.randn([A.ndim,A.ndim])
            >>> assert Matrix(A.KLdiv(B)) == (A*m).KLdiv(B*m)

        Can be created by re-arranging other functions

            >>> assert Matrix(A.KLdiv(B)) == A.entropy(B) - B.entropy()

            >>> b=B.sample(100)           
            >>> assert Matrix(A.entropy(b) - Mvn.fromData(b).entropy()) == A.KLdiv(b)

        What does this mean? shared information?:

            >>> assert Matrix(A.diag().KLdiv(A)) == A.diag().entropy() - A.entropy() 
            
#I'm not sure this is worth the backwards API.
        
        And calculating it for an Mvn is equivalent to averaging out a bunch of samples

            >>> N=1000
            >>> Z=3
            >>> 
            >>> #KLdiv 
            >>> a=A.sample(N)
            >>> KL = Mvn.fromData(A.KLdiv(a)) #should be zero
            >>> KL.var/=N
            >>> assert KL.mah2(0) < (Z**2)
            >>> 
            >>> #KLdiv 
            >>> b=B.sample(N)   
            >>> KL = Mvn.fromData(A.KLdiv(b) - A.KLdiv(B)) #difference should be zero
            >>> KL.var/=N
            >>> assert KL.mah2(0) < (Z**2)
            >>> 
            >>> B2=B.copy(deep=True)
            >>> B2.var/=2
            >>> b2=B2.sample(N)
            >>> KL = Mvn.fromData(B.KLdiv(b2) - B.KLdiv(B2)) #should be zero
            >>> KL.var/=N
            >>> assert KL.mah2(0) < (Z**2)

        http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        """
        if base is None:
            base=self.infoBase
#todo: multimethod?
        if isinstance(other,Mvn):
            det = abs(self.det())
            if det:
                baseE= (
                    (other/self).trace()+
                    ((self**-1)*(self.mean-other.mean).H).cov-
                    numpy.log(abs(other.pdet()/self.pdet()))-
                    self.rank
                )/2
                return (baseE/numpy.log(base))[0,0]
            else: 
                return numpy.inf 
        else:
            baseE= self.entropy(other)-type(self).fromData(other).entropy()            
            return (baseE/numpy.log(base))

    @decorate.prop
    class corners(object):
        """
        Get an iterator over the corners of the eigen-ellipse
        The points are placed at 1 standard deviations so that the matrix 
        has the same variance as the source

        The result is 2**rank, 1 x ndim vectors, that in total have the same 
        properties as the mvn they were pulled from:
            
            >>> if A.rank < 10:
            ...     C = Matrix([row for row in A.corners])
            ...     assert C.shape[0] == 2**A.rank
            ...     assert A == C 
            ...     assert A*M == C*M

        see also: X
        """
        def fget(self):
            scaled = self.scaled
            mean = self.mean
            rank = self.rank

            for n in range(2**self.rank):
                B=bin(n).split('b')[1]
                B='0'*(rank-len(B))+B
                positive = numpy.array([b=='1' for b in B])
                yield numpy.squeeze(
                    (scaled[positive,:].sum(0)-scaled[~positive,:].sum(0))
                    +mean
                )

    @decorate.prop
    class X(object):
        """
        Get the 'X' of points on the tips of the eigenvectors
        The points are placed at self.rank**(0.5) standard deviations so that the matrix 
        has the same variance as the source

        The result is a (2*rank x ndim) matrix that has the same properties as 
        the mvn it was pulled from:
            
            >>> assert isinstance(A.X,Matrix)
            >>> assert A.X.shape == (A.rank*2,A.ndim)
            >>> assert A==Mvn(A.X)
            >>> assert A*M == Mvn(A.X*M)
        """
        def fget(self):
            scaled = (self.rank**0.5)*self.scaled
            return numpy.vstack([scaled,-scaled])+self.mean


    ################# Non-Math python internals
    def __iter__(self):
        """
        iterate over the vectors in X so:
            >>> assert A == numpy.array([row for row in A])
        """
        return iter(numpy.squeeze(self.X))
    
    def __repr__(self):
        """
        print self
        
        assert A.__str__() == A.__repr__()
        
        assert A == eval(str(A))
        """
        return '\n'.join([
            '%s(' % self.__class__.__name__,
            '    mean=',
            '        %s,' % self.mean.__repr__().replace('\n','\n'+8*' '),
            '    var=',
            '        %s,' % self.var.__repr__().replace('\n','\n'+8*' '),
            '    vectors=',
            '        %s' % self.vectors.__repr__().replace('\n','\n'+8*' '),
            ')',
        ])
        
    
    __str__=__repr__

    ################ Art

    plotParams = {
        'edgecolor':'k'
    }
    """
    default plot parameters
    this could probably be broken up by number of dimensions in the plot,
    and by the number of dimensions of the object being     
    """
    
    
    def plot(self, ax=None, **kwargs):
        """
        :param ax:
        :param ** kwargs:
        """
        
        defaults = self.plotParams.copy()
        defaults.update(kwargs)
        
        return self._plotters.get(self.ndim,self.plotND)(self,ax,**defaults)
        
    def plot1D(self, 
        ax = None, 
        count = 1.0,
        fill = True, 
        nstd = 5,
        nsteps = 500, 
        orientation = 'horizontal',
        **kwargs
    ):
        """
        :param ax:
        :param count:
        :param fill:
        :param nstd:
        :param nsteps:
        :param orientation:
        :param ** kwargs:
        """                
        xlims = self.bBox(nstd).squeeze()
        x = numpy.linspace(xlims[0],xlims[1],nsteps)
        y = count*self.density(x[:,None])
        
        horizontal = ['horizontal','h','H']
        vertical   = ['vertical'  ,'v','V']
        
        if ax is None:
            ax=pylab.gca()        

        
        if fill:
            if orientation in horizontal:
                (xx,yy) = (y,x)
                filler = ax.fill_betweenx
            elif orientation in vertical:
                (xx,yy) = (x,y)
                filler = ax.fill_between
            else:
                raise ValueError(
                    'orientation should be in either (%s) or (%s), not (%s)' % 
                    (horizontal,vertical,orientation)
                )
            plotter = functools.partial(filler,x,y,0)
        else:
            plotter = functools.partial(ax.plot,x,y)
        
        return plotter(**kwargs)
                
    def plot2D(self,ax = None, nstd = 2, **kwargs):
        """
        :param ax:
        :param nstd:
        :param ** kwargs:
            
        plot a :py:meth:`mvn.Mvn.patch`, with axis autoscaling
        """
        if ax is None:
            ax=pylab.gca()
 
        bBox = self.bBox(nstd).array()
        widths = numpy.diff(bBox,axis=0)
        pads = 0.05*widths*[[-1],[1]]
        corners = bBox+pads
        ax.update_datalim(corners)
        ax.autoscale_view()
                        
        artist = self.patch(**kwargs)
        
        if isinstance(artist,matplotlib.lines.Line2D):
            insert = ax.add_line
        else:
            insert = ax.add_patch
            
        return insert(artist)
        
    def plot3D(self,ax = None, nstd = 2.0, **kwargs):
        """
        :param ax:
        :param ** kwargs:
        """
        from mpl_toolkits.mplot3d import Axes3D

        if ax is None:
            ax = pylab.gca(projection = '3d')
            
        assert isinstance(ax,Axes3D)  
        
        
        u = numpy.linspace(0, 2 * numpy.pi, 100)
        v = numpy.linspace(0, numpy.pi, 100)

        x = numpy.outer(numpy.cos(u), numpy.sin(v))
        y = numpy.outer(numpy.sin(u), numpy.sin(v))
        z = numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

        xyz =[x[...,None],y[...,None],z[...,None]]
    
        xyz = numpy.concatenate(xyz,-1)*nstd
        
        xyz = numpy.dot(
            xyz,
            self.scaled.array()
        )

        xyz = xyz+self.mean.array()[None,:,:]
        
        ax.plot_wireframe(
            xyz[...,0], xyz[...,1], xyz[...,2],  
            rstride=10, 
            cstride=10, 
            color = 'k',
            **kwargs
        )
            
        
    def plotND(self,ax = None, **kwargs):
        """
        :param ax:
        :param ** kwargs:
        """
        raise NotImplementedError()
        
    _plotters={1:plot1D,2:plot2D,3:plot3D}
    """
    >>> assert mvn._plotters[1] is mvn.plot1D
    >>> assert mvn._plotters[2] is mvn.plot2D
    >>> assert mvn._plotters[3] is mvn.plot3D    
    """    
    
#multimethod distributor
    def patch(self,nstd=2,alpha='auto',slope=0.5,minalpha=0.3,**kwargs):
        """
        :param nstd:
        :param alpha:
        :param slope:
        :param minalpha:
        :param ** kwargs:
            
        get a matplotlib Ellipse patch representing the Mvn, all \**kwargs are 
        passed on to the call to matplotlib.patches.Ellipse

        not surprisingly Ellipse only works for 2d data.

        the number of standard deviations, 'nstd', is just a multiplier for 
        the eigen values. So the standard deviations are projected, if you 
        want volumetric standard deviations I think you need to multiply by 
        sqrt(ndim)

        if  you don't specify a value for alpha it is set to the exponential of 
        the area, as if it has a fixed amount if ink that is spread over it's area.

        the 'slope' and 'minalpha' parameters control this auto-alpha:
            'slope' controls how quickly the the alpha drops to zero
            'minalpha' is used to make sure that very large elipses are not invisible.  
        """
        shape = self.shape

        if shape[1] != 2:
            raise ValueError(
                'this method can only produce patches for 2d data'
            )
        
        if shape[0]<2:
            facecolor = kwargs.pop('facecolor',None)
            if facecolor is not None:
                kwargs['markerfacecolor'] = facecolor
                
            edgecolor = kwargs.pop('edgecolor',None)
            if edgecolor is not None:
                kwargs['markeredgecolor'] = edgecolor
                kwargs['color'] = edgecolor
                
            if 'marker' not in kwargs:
                kwargs['marker'] = 'o'            
            
            if shape[0] == 0:
                return matplotlib.lines.Line2D(self.mean[:,0],self.mean[:,1],**kwargs)
            elif shape[0] == 1:
                delta=nstd*self.scaled
                front=self.mean+delta
                back=self.mean-delta
                data = numpy.vstack([front,back])
                    
                return matplotlib.lines.Line2D(data[:,0],data[:,1],**kwargs)
            

        if alpha=='auto':

            alpha = numpy.max([
                minalpha,
                numpy.exp(-slope*sqrt(self.det()))
            ])

            colorConverter = matplotlib.colors.ColorConverter()           
            
            facecolor = kwargs.get('facecolor')
            if facecolor is not None:
                if isinstance(facecolor,str):
                    facecolor = colorConverter.to_rgb(facecolor)
                facecolor = list(facecolor)
                if len(facecolor) < 4:
                    facecolor.append(alpha)
                kwargs['facecolor'] = facecolor

            edgecolor = kwargs.get('edgecolor')
            if edgecolor is not None:
                if isinstance(edgecolor,str):
                    edgecolor = colorConverter.to_rgb(edgecolor)
                edgecolor = list(edgecolor)
                if len(edgecolor) < 4:
                    edgecolor.append(alpha)
                kwargs['edgecolor'] = edgecolor

        #unpack the width and height from the scale matrix 
        wh = nstd*sqrt(self.var)
        wh[wh>1e5]=1e5

        #convert from radius to diameters
        width,height=2*wh

        #return an Ellipse patch
        return matplotlib.patches.Ellipse(
            #with the Mvn's mean at the centre 
            xy=tuple(self.mean.flatten()),
            #matching width and height
            width=width, height=height,
            #and rotation angle pulled from the vectors matrix
            angle=180/numpy.pi*(numpy.angle(helpers.ascomplex(self.vectors)[0,0])),
            #while transmitting any kwargs.
            **kwargs
        )
        

        


## extras    

def wiki(P,M):
    """
    :param P:
    :param M:
        
    Direct implementation of the wikipedia blending algorithm
    
    The quickest way to prove it's equivalent is by examining these:

    >>> if not (A.flat or B.flat):    
    ...     assert A**-1 == A*A**-2
    ...     assert A & B == (A*A**-2+B*B**-2)**-1
    ...
    ...     D = A*(A.cov)**(-1) + B*(B.cov)**(-1)
    ...     assert wiki(A,B) == D*(D.cov)**(-1)
    ...     assert A & B == wiki(A,B)
    """
    yk=M.mean-P.mean
    Sk=P.cov+M.cov
    Kk=P.cov*Sk.I
    
    return Mvn.fromCov(
        mean=(P.mean + yk*Kk.H),
        cov=(Matrix.eye(P.ndim)-Kk)*P.cov
    )

def givenVector(self,dims,value):
    """
    :param dims:
    :param value:
    
    direct implementation of the "given" algorithm in
    Andrew moore's data-mining/gussian slides

    >>> assert givenVector(A,dims=0,value=1)==A.given(dims=0,value=1)
    """
    fixed=binindex(dims,self.ndim)
    if fixed.all():
        return Mvn.fromData(value)

    free=~fixed

    Mu = self[:,free]
    Mv = self[:,fixed]
    #todo: cleanup
    u=self.vectors[:,free]
    v=self.vectors[:,fixed]

    uv = numpy.multiply(u.H,self.var)*v

    result = Mu-(Mv-value)**-1*uv.H

    #create the mean, for the new object,and set the values of interest
    mean=numpy.zeros([1,self.ndim],dtype=result.mean.dtype)
    mean[:,fixed]=value
    mean[:,free]=result.mean

    #create empty vectors for the new object
    vectors=numpy.zeros([
        result.shape[0],
        self.ndim
    ],result.vectors.dtype)
    vectors[:,fixed]=0
    vectors[:,free]=result.vectors
    
    return type(self)(
        mean=mean,
        vectors=vectors,
        var=result.var
    )



def mooreChain(self,sensor,transform=None):
        """
        :param sensor:
        :param transform:
            
        given a distribution of actual values and an Mvn to act as a sensor 
        this method returns the joint distribution of real and measured values

        the, optional, transform parameter describes how to transform from actual
        space to sensor space
        """

        if transform is None:
            transform=Matrix.eye(self.ndim)

        T=(self*transform+sensor)
        vv=self.cov        

        return type(self).fromCov(
            mean=numpy.hstack([self.mean,T.mean]),
            cov=numpy.vstack([
                numpy.hstack([vv,vv*transform]),
                numpy.hstack([(vv*transform).H,T.cov]),
            ])
        )

def binindex(index,size):
    """
    :param index:
    :param size:
        
    convert an index to binary so it can be easily inverted
    """
    if hasattr(index,'dtype') and index.dtype==bool:
        return index
    
    binindex=numpy.zeros(size,dtype=bool)
    binindex[index]=True

    return binindex



if __name__ == '__main__':
    print 'hello'

    #overwrite everything we just created with the copy that was 
    #created when we imported mvn; there can only be one.
    from testObjects import *

   # B & A & B == A

    print A  > Matrix.infs(2)

#    
#    A < -Matrix.infs(A.ndim)
#
#    import operator
#    M = [A.marginal(n) for n in range(A.ndim)]
#    assert reduce(operator.and_,M) == A.diag()
#    
#    assert Matrix([A[:,n].var[0] for n in range(A.ndim)]) == A.width()**2

#    AB  = A &  B
#    A_B = A & ~B
        
#    locations = AB.sample([10,10])

    # A&B == k*A.*B
#    Da = A.density(locations)
#    Db = B.density(locations)

#    Dab  = (AB).density(locations)

#    ratio = Dab /(Da*Db)
#    assert(Matrix(0) == ratio.var())   

    # A&(~B) == k*A./B
#    Da_b = (A_B).density(locations)
#    ratio = Da_b/(Da/Db)
#    assert(Matrix(0) == ratio.var())

    
#    A/B == A*(B**(-1))

#    b=B**0

#    assert b+b == 2*b

#    N1=1000
#    N2=10
#    data1 = numpy.random.randn(N1,2)+5*numpy.random.randn(1,2)
#    data2 = numpy.random.randn(N2,2)+5*numpy.random.randn(1,2)
    
#    A = Mvn.fromData(data1)
#    B = Mvn.fromData(data2)

#    print Mvn.fromData([A,B],Matrix.zeros) 
#    print Mvn.fromData(A,Matrix.zeros)+Mvn.fromData(B,Matrix.zeros)

