#! /usr/bin/env python

#todo: understand transforms composed of Mvars as the component vectors, and 
#          whether it is meaningful to consider mvars in both the rows and columns
#todo: implement a transpose,for the above 
#todo: see if div should be upgraded to act more like matlab backwards divide
#todo: impliment collectionss of mvars so that or '|'  is meaningful
#todo: cleanup my 'square' function (now that it is clear that it's an SVD)
#todo: entropy?
#todo: quadratic forms (ref: http://en.wikipedia.org/wiki/Quadratic_form_(statistics))
#todo: chain rule(see moore's datamining slides), it looks like a division 
#todo: start using unittest instead of just doctest
#todo: split the class into two levels: "fast" and 'safe'?
#      maybe have the 'safe' class inherit from 'fast' and a add a variance-free 'plane' class?
#todo: understand the EM and K-means algorithms (available in scipy)
#todo: understans what complex numbers imply with these.
#todo: understand the relationship betweent hese and a hessian matrix.
#todo: figure out the relationship between these and spherical harmonics
#todo: investigate higher order cumulants, 'principal cumulant analysis??'

"""
This module contains only two things: the "Mvar" class, and the "wiki" 
function.

Mvar is the main idea of the module: Multivariate normal distributions 
    packaged to act like a vector. Perfect for kalman filtering, sensor fusion, 
    (and maybe Expectation Maximization)  

wiki is just to demonstrate the equivalency between my blending algorithm, 
    and the wikipedia version of it.
        http://en.wikipedia.org/wiki/Kalman_filtering#Update

The docstrings are full of examples. The objects used in theexamples are created 
by test.sh, and stored in test_objects.pkl. You can get the most recent versions of them by 
importing testObjects.py, which will give you a module containing the objects used

in the tests
    A,B and C are instances of the Mvar class  
    K1 and K2 are random numbers
    M and M2 are matrixes
    E is an apropriately sized eye matrix
    N is an integer

remember: circular logic works because circluar logic works.
    a lot of the examples are demonstrations of what the code is doing, or expected
    invariants. they don't prove I'm right, but only that I'm being consistant
 
"""

############  imports

## builtins
import itertools
import collections 
import copy
import operator

## 3rd party
import numpy

#from scipy import sqrt()
def sqrt(data):
    """
    like scipy.sqrt without a scipy depandancy
    """
    data = numpy.asarray(data)
    if numpy.isreal(data).all() and (data>=0).all():
        return numpy.sqrt(data)
    return data**(0.5+0j)

## maybe imports: third party things that we can live without
from maybe import Ellipse

## local
import helpers

from square import square
from automath import Automath
from right import Right
from inplace import Inplace
from matrix import Matrix

class Mvar(Automath,Right,Inplace):
    """
    Multivariate normal distributions packaged to act like a vector 
    (Ref: andrew moore / data mining / gaussians )
    (Ref: http://en.wikipedia.org/wiki/Vector_space)
    
    The class fully supports complex numbers.
    
    basic math operators (+,-,*,/,**,&) have been overloaded to work 'normally'
        for kalman filtering and common sense. But there are several surprising 
        features in the math these things produce, so watchout. (It basically 
        boils down to the fact that there are differences between scalar, matrix, 
        and mvar operations, and if re-aranging an equation changes which type of 
        operation is called, the results will be different.)
    
    This is perfect for kalman filtering, sensor fusion, or anything where you 
        need to track linked uncertianties across multiple variables 
        like expectation maximization principal component analysis 
        (ref: http://en.wikipedia.org/wiki/Expectation-maximization_algorithm)

    kalman filtering simplifies to: 
        
        state[t+1] = (state[t]*STM + noise) & measurment
        
        Where 'state' is a series of mvars (indexed by time), 'noise' and 
        'measurment' are Mvars, ('noise' having a zero mean) and 'STM' is the 
        state transition matrix
        
    Sensor fusion is just:
        result = measurment1 & measurrment2 & measurment3
        
    The data is stored as mean, variance, and (eigen)vectors 
        but other useful info in accessable through virtual attributes 
        (properties).
    
        This system makes 'compression' (like principal component analysis, or 
        thin SVD) easy and maybe more useful, since I can calculate the eigenvectors 
        without necessarily calculating a full covariance matrix
    
    Attributes:
        mean
            mean of the distribution
        var
            the variance asociated with each vector.
        vectors
            unit vectors, as rows, ?not necessarily orthogonal?. 
            only guranteed to give the right covariance see below.
        
    Properties:
        
        ndim
            >>> assert A.ndim == A.mean.size

        cov
            get or set the covariance matrix
            >>> assert A.vectors.H*numpy.diagflat(A.var)*A.vectors == A.cov
    
        scaled
            gets the vectors, scaled by one standard deviation
            (transforms from unit-eigen-space to data-space) 
            
        transform
            >>> assert A.transform()**2 == abs(A).cov 
            >>> if not(flat and N<0):
            ...     assert A.transform()**N == A.transform(N)

            >>> assert A.transform(2) == abs(A).cov
            
            this is just more efficient than square-rooting the covariance matrix, 
            since it is stored de-composed
            (transforms from unit-data-space to data-space) 
            
    
    The from* functions all create new instances from varous 
    common constructs.
        
    The get* functions all grab useful things out of the structure
    
    The inplace operators (like +=) work, inplace, but unlike in many classes, 
    do not currently speed up any operations.
    
    The mean of the distribution is stored as a row vector, so make sure to align 
    your transforms apropriately and have the Mvar on left the when attempting 
    to do a matrix multiplies on it. This is for two reasons: 

        1) inplace operators work nicely (Mvar on the left)
        
        2) The Mvar is (currently) the only object that knows how to do 
        operations on itself, might as well go straight to it instead of 
        passing around "NotImplemented" 
        
    No work has been done to make things fast, because until they work at all 
    speed is not worth working on.  
    """
    
    ############## Creation
    def __init__(self,
        vectors=Matrix.eye,
        var=numpy.ones,
        mean=numpy.zeros,
        square=True,
        squeeze=True,
        **kwargs
    ):
        """
        self(**attributes)

        Create an Mvar from available attributes.
        
        vectors: defaults to zeros
        var: (variance) defaults to ones
        
        >>> assert A.vectors.H*numpy.diagflat(A.var)*A.vectors == A.cov
        
        mean: defaults to zeros
        
        square:
            if true squares up the self before returning it. This sets the 
            vectors to orthogonal and unit length.
            
        squeeze:
            calls self.squeeze() on the result if true. To clear out any 
            low valued vectors. It uses the same defaults as numpy.allclose()

        **kwargs is only used to pass in non standard defaults to the call to 
            squeeze, which is similar to numpy.allclose, 
            defaults are rtol=1e-5, atol=1e-8
        """
        #stack everything to check sizes and automatically inflate any 
        #functions that were passed in
        
        var= var if callable(var) else numpy.array(var).flatten()[:,None]
        mean= mean if callable(mean) else numpy.array(mean).flatten()[None,:]
        vectors= vectors if callable(vectors) else Matrix(vectors)
        
        stack=Matrix(helpers.autostack([
            [var,vectors],
            [1  ,mean   ],
        ]))
        
        #unpack the stack into the object's parameters
        self.mean = numpy.real_if_close(stack[-1,1:])
        self.var = numpy.real_if_close(numpy.array(stack[:-1,0]).flatten())
        self.vectors = numpy.real_if_close(stack[:-1,1:])
        
        if square:
            self.copy(self.square())

        if squeeze:
            self.copy(self.squeeze(**kwargs))

    ############## alternate creation methods
    @staticmethod
    def fromCov(cov,**kwargs):
        """
        everything in kwargs is passed directly to the constructor
        """
        diag = Matrix(numpy.diag(cov))
        eig = numpy.linalg.eigh if abs(diag) == diag else numpy.linalg.eig
        #get the variances and vectors.
        (var,vectors) = eig(cov) if cov.size else (Matrix([]),Matrix([]))
        vectors=Matrix(vectors.H)     

        return Mvar(
            vectors=vectors,
            var=var,
            square=False,
            **kwargs
        )
    
    @staticmethod
    def fromData(data,mean=None,weights=None, bias=True, **kwargs):
        """
        >>> assert Mvar.fromData(A)==A 
        
        >>> data=[1,2,3]
        >>> new=Mvar.fromData(data)
        >>> assert new.mean == data
        >>> assert (new.var == numpy.zeros([0])).all()
        >>> assert new.vectors == numpy.zeros([0,3])
        >>> assert new.cov == numpy.zeros([3,3])
        
        bias is passed to numpy's cov function.
        
        any kwargs are just passed on the Mvar constructor.
        
        this creates an Mvar with the same mean and covariance as the supplied 
        data with each row being a sample and each column being a dimenson
        
        remember numpy's default covariance calculation divides by (n-1) not 
        (n) set bias = 1 to use N,
        """
        
        if isinstance(data,Mvar):
            return data.copy()
        
        data=Matrix(data)

        
        #todo: implement these
        assert weights is None,'weights not implemented'
        assert data.dtype is not numpy.dtype('object'),'not mplementd for mvars yet'
        
        #get the number of samples, subtract 1 if un-biased
        N=data.shape[0] if bias else data.shape[0]-1
        
        #get the mean of the data
        if mean is None:
            mean=numpy.mean(data,axis=0)

        if weights is None:
        
            cov=(data.H*data)/N-mean.H*mean
        
            #create the mvar from the mean and covariance of the data
            return Mvar.fromCov(
                cov = cov,
                mean= mean,
                **kwargs
            )
    
    @staticmethod
    def zeros(n=1):
        """
        >>> if N>0:
        ...     Z=Mvar.zeros(N)
        ...     assert Z.mean==Matrix.zeros
        ...     assert Z.var.size==0
        ...     assert Z.vectors.size==0
        """
        return Mvar(mean=Matrix.zeros(n))
    
    @staticmethod
    def infs(n=1):
        """
        >>> if N>0:
        ...     inf=Mvar.infs(N)
        ...     assert inf.mean==Matrix.zeros
        ...     assert inf.var.size==inf.mean.size==N
        ...     assert (inf.var==numpy.inf).all()
        ...     assert inf.vectors==Matrix.eye
        """
        
        return Mvar.zeros(n)**-1
    
    ##### 'cosmetic' manipulations
    def inflate(self):
        """
        add the zero length direction vectors so no information is lost during 
        rotations

        >>> if A.shape[0] == A.shape[1]:
        ...     assert A*A.vectors.H*A.vectors==A
        
        >>> if A.shape[0] != A.shape[1]:
        ...     assert A*A.vectors.H*A.vectors!=A

        >>> A=A.inflate()
        >>> assert A*A.vectors.H*A.vectors==A        
        """
        result = self.copy()

        shape=self.shape        

        missing = shape[1]-shape[0]

        if missing == 0:
            return result
        elif missing<0:
            return result.square()


        result.var = numpy.concatenate([result.var,numpy.zeros(missing)])
        
        result.vectors = numpy.vstack(
            [self.vectors,numpy.zeros((missing,shape[1]))]
        )

        result = result.square()

        zeros=helpers.approx(result.var)

        result.var[zeros]=0

        return result

    def squeeze(self,**kwargs):
        """
        drop any vector/variance pairs with (self.var) under the tolerence limit
        the default tolerence is 1e-12,
        """
        result=self.copy()
        
        small=helpers.approx(self.var,**kwargs)
        
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
        result=self.copy()
        (result.var,result.vectors)=square(
            vectors=self.vectors,
            var=self.var,
        )
        return result

    
    ############ setters/getters -> properties
    
    cov = property(
        fget=lambda self:self.vectors.H*numpy.diagflat(self.var)*self.vectors, 
        fset=lambda self,cov:self.copy(
            Mvar.fromCov(
                mean=self.mean,
                cov=cov,
        )),
        doc="""
            get or set the covariance matrix used by the object
        
            >>> assert A.cov==A.vectors.H*numpy.diagflat(A.var)*A.vectors
            >>> assert abs(A).cov==A.scaled.H*A.scaled
        """
    )
    
    scaled = property(
        fget=lambda self:Matrix(numpy.diagflat(sqrt(self.var)))*self.vectors,
        doc=
        """
        get the vectors, scaled by the standard deviations. 
        Useful for transforming from unit-eigen-space, to data-space

        >>> assert A.vectors.H*A.scaled==A.transform()
        """
    )

    def transform(self,power=1):
        """
            >>> assert A.transform() == A.transform(1)
            
            >>> assert A.cov == (A**2).transform()
            >>> assert A.cov == A.transform()*A.transform()
            >>> assert A.cov == A.transform()**2
            >>> assert A.cov == A.transform(2)
            
            >>> assert A.transform(N)== (A**N).transform()
            >>> assert A.transform(N)== A.transform()**N  
            >>> #it's hit and miss for complex numbers, but real is fine
            >>> assert (A**K1.real).transform() == A.transform(K1.real) 

            >>> assert (A*B.transform() + B*A.transform()).cov/2 == (A*B).cov

            >>> assert Matrix(numpy.trace(A.transform(0))) == A.shape[0] 
        """
        if not numpy.isreal(self.var).all() or not(self.var>0).all():
            power = complex(power)

        if helpers.approx(power):
            vectors=self.vectors
            varP=numpy.ones_like(self.var)
        else:
            #the null vectors are automatically being ignored,
            #ignore the infinite ones as well
            keep=~helpers.approx(self.var**(-1))
            
            varP=self.var[keep]**(power/2.0)
            vectors=self.vectors[keep,:]

        return (
            vectors.H*
            numpy.real_if_close(numpy.diagflat(varP))*
            vectors
        )

    ndim=property(
        fget=lambda self:(self.mean.size),
        doc="""
            get the number of dimensions of the space the mvar exists in
            >>> assert A.ndim==A.mean.size
        """
    )
    
    shape=property(
        fget=lambda self:(self.vectors.shape),
        doc="""
            get the shape of the vectors,the first element is the number of 
            vectors, the second is their lengths: the number of dimensions of 
            the space they are embedded in
            
            >>> assert A.vectors.shape == A.shape
            >>> assert (A.var.size,A.mean.size)==A.shape
            >>> assert A.shape[1]==A.ndim
        """
    )

    def sign(self):
        return sign(self.var)

    ########## Utilities
        
    @staticmethod
    def stack(*mvars,**kwargs):
        """
        >>> AB= Mvar.stack(A,B)
        >>> assert AB[:A.ndim]==A
        >>> assert AB[A.ndim:]==B

        It's a static method to make it clear that this is not happening in place
        
        Stack two Mvars together, equivalent to hstacking the means, and 
        diag-stacking the covariances
        
        yes it works but be careful. Don't use this for reconnecting 
        something you calculated from an Mvar, back to the same Mvar it was 
        calculated from, you'll loose all the cross corelations. 
        If you're trying to do that use a better matrix multiply. 
        
        is there a connection between this and 'chain'? (ref: andrew moore/data mining/gaussians)
        """
        #no 'square' is necessary here because the rotation matrixes are in 
        #entierly different dimensions
        return Mvar(
            #stack the means
            mean=numpy.hstack([mvar.mean for mvar in mvars]),
            #stack the vector diagonally
            vectors=helpers.diagstack([mvar.vectors for mvar in mvars]),
            var=numpy.concatenate([mvar.var for mvar in mvars]),
            **kwargs
        )
    
    def sample(self,n=1,cplx=False):
        """
        take samples from the distribution
        n is the number of samples, the default is 1
        each sample is a numpy matrix row vector.
        
        a large number of samples will have the same mean and cov as the 
        Mvar being sampled
        """
        units = Matrix(
            helpers.ascomplex(numpy.random.randn(n,self.ndim,2))/sqrt(2)
            if cplx else 
            numpy.random.randn(n,self.ndim)
        )
        return Matrix(numpy.array(units*self.scaled.T)+self.mean)

    def measure(self,actual):
        """
        This method is to simulate a sensor. 
     
        It treats the Mvar as a the description of a sensor, 
        the mean is the sensor's bias, and the variance is the sensor's variance.

        The result is an Mvar, with the mean being a sample pulled from the sensor's 
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
        """
        sample=self.sample(1)-self.mean
        sample[numpy.isnan(sample)]=0

        return self+(sample+actual)

    def det(self):
        """
        returns the determinant of the covariance matrix. 
        this method is supplied because the determinant can be calculated 
        easily from the variances in the object
        
        >>> assert Matrix(A.det())== numpy.linalg.det(A.cov)
        >>> assert Matrix(A.det()) == (
        ...     0 if 
        ...     A.shape[0]!=A.shape[1] else 
        ...     numpy.prod(A.var)
        ... )
        """
        shape=self.shape
        return (
            0 if 
            shape[0]!=shape[1] else 
            numpy.prod(self.var)
        )
        
    def trace(self):
        """
        returns the trace of the covariance matrix.
        this method is supplied because the trace can be calculated 
        easily from the variances in the object
        
        >>> assert Matrix(A.trace()) == numpy.trace(A.cov)
        >>> assert Matrix(A.trace()) == A.var.sum()
        """
        return self.var.sum()
    
    def quad(self,matrix):
        """
        place holder for quadratic forum
       
        """
        #todo: implement quadratic forms?
        assert 1==0

    def entropy(self):
        """
        place holder for entroppy function...
        """
        #todo: impliment entropy function
        assert 1==0
        
    def chain(self,other):
        """
        place holder for chain (ref: andrew moore/data mining/gaussians)
        """
        #todo: impliment entropy function
        assert 1==0
        
    def dist2(self,locations):
        """
        return the square of the mahalabois distance from the Mvar to each vector.
        the vectors should be along the last dimension of the array.

        >>> assert helpers.approx(
        ...     (A**0).dist2(numpy.zeros((1,ndim))),
        ...     helpers.mag2((A**0).mean)
        ... ) or flat
        """
        #make sure the mean is a flat numpy array
        mean=numpy.array(self.mean).squeeze()
        #and subtract it from the locations (vectors aligned to the last dimension)
        locations=numpy.array(locations)-mean
        #rotate each location to eigen-space
        rotated=numpy.dot(locations,numpy.array(self.vectors.H))
        #get the square of the magnitude of each component
        squared=rotated.conjugate()*rotated
        #su over the last dimension
        return numpy.real_if_close((
                squared*(self.var**-1)
            ).sum(axis = locations.ndim-1)
        )
        
    ############## indexing
    
    def given(self,index,value):
        """
        return an mvar representing the conditional probability distribution, 
        given the values, on the given indexes
        
        I think this equivalent to: andrew moore/data mining/gussians/page 22
        
        basic usage fixes the indexed component of the mean to the given value 
        with zero variance in that dimension.
        
        >>> a = A.given(index=0,value=1)
        >>> assert a.mean[:,0]==1
        >>> assert a.vectors[:,0]==numpy.zeros

        this equivalent to doing an __and__ with an mvar of the apropriate shape
        zero var on the indexed dimensions, infinite vars on the others
        
        >>> L1=Mvar(mean=[0,0],vectors=[[1,1],[1,-1]], var=[numpy.inf,0.5])
        >>> L2=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf) 
        >>> assert L1.given(index=0,value=1) == L1&L2
        >>> assert (L1&L2).mean==[1,1]
        >>> assert (L1&L2).cov==[[0,0],[0,2]]
        
        the above examples are with scalars but vectors work with apropriate 
        indexes
        
        because this is just an interface to __and__ the logical extension 
        becomes obvious:
        
        >>> Y=Mvar(mean=[0,1],vectors=Matrix.eye, var=[numpy.inf,1])
        >>> X=Mvar(mean=[1,0],vectors=Matrix.eye,var=[1,numpy.inf])
        >>> x=Mvar(mean=1,var=1)
        >>> assert Y.given(index=0,value=x) == X&Y
        
        __setitem__ uses this for an inplace version
        
        >>> a=A.copy()
        >>> a[0]=1
        >>> assert a==A.given(index=0,value=1)
        """
        #convert the inputs
        value=Mvar.fromData(value)
        
        #create the mean, for the new object,and set the values of interest
        mean=numpy.zeros([1,self.shape[0]])
        mean[0,index]=value.mean

        #create empty vectors for the new object
        vectors=numpy.zeros([
            value.shape[0]+(self.ndim-value.ndim),
            self.ndim
        ])
        vectors[0:value.shape[0],index]=value.vectors
        vectors[value.shape[0]:,~index]=numpy.eye(self.ndim-value.ndim)
        
        #create the variance for the new object
        var=numpy.zeros(vectors.shape[0])
        var[0:value.shape[0]]=value.var
        var[value.shape[0]:]=numpy.Inf

        #blend with the self
        return self & Mvar(
            var=var,
            mean=mean,
            vectors=vectors,
        ) 
        
    def __setitem__(self,index,value):
        """
        self[index]=value
        
        this is an opertor interface to self.given 
        
        watchout, this only takes a single index, not two like __getitem__
        the asymetry is unavoidable, they do very different things 
        """
        self.copy(self.given(index,value))

    def __getitem__(self,index):
        """
        self[index]
        return the marginal distribution,
        over the indexed dimensions,
        """
        return Mvar(
            mean=self.mean[:,index],
            vectors=self.vectors[:,index],
            var=self.var,
        )

    ############ Math

    def __eq__(self,other):
        """
        self == other

        mostly it does what you would expect

        >>> assert A==A.copy()
        >>> assert A is not A.copy()
        >>> assert A != B

        You'll get an Error if the mvars have different numbers of dimensions. 
        
        Infinite, and zero variances are handled correctly.
        
        Note that the component of the mean along a direction with infinite variance is ignored:

            >>> assert (
            ...     Mvar(mean=[1,0,0], vectors=[1,0,0], var=numpy.inf)==
            ...     Mvar(mean=[0,0,0], vectors=[1,0,0], var=numpy.inf)
            ... )

        __ne__ is handled by the Automath class

            >>> assert A != B

        """
        other=Mvar.fromData(other)
        
        #check the number of dimensions of the space
        assert (
            self.ndim == other.ndim,
            """
            if the objects have different numbers of dimensions, 
            you're doing something wrong
            """
        )

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
   
        H=lambda M:M.H*M

        SFvectors=self.vectors[Sfinite]
        SFvar=self.var[Sfinite]

        OFvectors = other.vectors[Ofinite]
        OFvar = other.var[Ofinite]

        cov=lambda vectors,var: vectors.H*numpy.diagflat(var)*vectors

        #compare the finite and infinite covariances 
        return (
            cov(SFvectors,SFvar) == cov(OFvectors,SFvar) and
            SIvectors.H*SIvectors == OIvectors.H*OIvectors
        )
        
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
        except if the mvar is flat; that flattness is preserved 
        (because zeros get squeezed, positive and negative zeros are indistinguishable)

        >>> assert (A & ~A) == Mvar(mean=numpy.zeros(A.ndim))**-1 or flat
        >>> assert (A & ~A) == Mvar(mean=A.mean, vectors=A.vectors, var=Matrix.infs)

        infinite variances provide no information, having a no effect when blended

        >>> assert A == A & (B & ~B) or flat
        
        if the mvar is flat, things are a little different:
            like this you're taking a slice of A in the plane of B
            >>> assert  A &(B & ~B) == A & Mvar(mean=B.mean, vectors=B.vectors, var=Matrix.infs)
   
            but watch out:
            >>> assert (A&~B) & B == (A&B) & ~B
            >>> (A&B) & ~B == A & (B&~B) and flat
            False

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

        the automath logic extensions are actually useless to Mvar because:
            >>> assert (~A & ~B) == ~(A & B)

            so 'or' would become a copy of 'and' and 'xor' would become a blank equavalent to the (A & ~A) above

            maybe the A|B = A+B - A&B  version will be good for something; I'll put them in for now
        """
        result=self.copy()
        result.var=-(self.var)
        return result
    
    def __or__(self,other):
        """
        self | other
        I don't  know what this means yet
        """
        #todo: create a 'GMM' class so that | has real meaning
        return self+other-self&other

    def __xor__(self,other):
        """
        I don't  know what this means yet
        """
        return self+other-2*(self&other)

    def __and__(self,other):
        """
        self & other
        
        This is awsome.
        
        it is the blend step from kalman filtering
        
        optimally blend together two mvars, this is done with the 
        'and' operator because the elipses look like ven-diagrams
        
        Just choosing an apropriate inversion operator (1/A) allows us to 
        define kalman blending as a standard 'paralell' operation, like with 
        resistors. operator overloading takes care of the rest.
        
        The inversion automatically leads to power, multiply, and divide  
        
        >>> assert A & B == B & A 
        >>> assert A & B == 1/(1/A+1/B) or flat
        
        >>> abc=numpy.random.permutation([A,B,C])
        >>> assert A & B & C == helpers.paralell(*abc) or flat
        >>> assert A & B & C == reduce(operator.and_ ,abc) or flat
        
        >>> assert (A & B) & C == A & (B & C) or flat
        
        >>> assert (A & A).cov == A.cov/2
        >>> assert (A & A).mean == A.mean
        
        >>> assert A &-A == Mvar(mean=numpy.zeros(ndim))**-1 or flat
        >>> assert A &~A == Mvar(mean=numpy.zeros(ndim))**-1 or flat
        
        The proof that this is identical to the wikipedia definition of blend 
        is a little too involved to write here. Just try it (and see the "wiki"
        function)
        
        >>> assert A & B == wiki(A,B) or flat

        this algorithm is also, at the same time, solving linear equations
        where the zero vatiances correspond to a plane's null vectors 

        >>> L1=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf)
        >>> L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[1,1]
        >>> assert (L1&L2).var.size==0

        >>> L1=Mvar(mean=[0,0],vectors=[1,1],var=numpy.inf)
        >>> L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[1,1]
        >>> assert (L1&L2).var.size==0
        
        >>> L1=Mvar(mean=[0,0],vectors=Matrix.eye, var=[1,1])
        >>> L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[0,1]
        >>> assert (L1&L2).var==1
        >>> assert (L1&L2).vectors==[1,0]
        
    """
        #check if they both fill the space
        if (
            self.mean.size == (self.var!=0).sum() and 
            other.mean.size == (other.var!=0).sum()
        ):
            #then this is a standard paralell operation
            return (self**(-1)+other**(-1))**(-1) 
        
        #otherwise there is more work to do
        
        #inflate each object
        self=self.inflate()
        other=other.inflate()
        #collect the null vectors
        Nself=self.vectors[self.var==0,:]
        Nother=other.vectors[other.var==0,:] 

        #and stack them
        null=numpy.vstack([
            Nself,
            Nother,
        ])

        #get length of the component of the means along each null vector
        r=numpy.hstack([self.mean*Nself.H,other.mean*Nother.H])

        #square up the null vectors
        (s,v,d)=numpy.linalg.svd(null,full_matrices=False)

        #discard any very small components
        nonZero = ~helpers.approx(v**2)
        s=s[:,nonZero]
        v=v[nonZero]
        d=d[nonZero,:]
        
        #calculate the mean component in the direction of the new null vectors
        Dmean=r*s*numpy.diagflat(v**-1)*d
        
        #do the blending, while compensating for the mean of the working plane
        return ((self-Dmean)**-1+(other-Dmean)**-1)**-1+Dmean

    def __pow__(self,power):
        """
        self**power

        >>> #the transform version doesn't work for flat objects if the transform power is less than 0
        >>> k = K1.real
        >>> if not flat or k>0:
        ...     assert A**k == A*A.transform(k-1) + Mvar(mean=A.mean-A.mean*A.transform(0)) 

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
            >>> assert A == (A**-1)**-1 or flat
            >>> assert A.mean*A.transform(0) == ((A**-1)**-1).mean
    
            >>> assert A**0*A==A
            >>> assert A*A**0==A
            >>> if not flat:
            ...     assert A**0 == A**(-1)*A
            ...     assert A**0 == A*A**(-1)
            ...     assert A**0 == A/A 
            ...     assert A/A**-1 == A**2
            
            >>> False if flat else (A**K1)*(A**K2)==A**(K1+K2)
            False

            >>> False if flat else A**K1/A**K2==A**(K1-K2)
            False

            those only work if the k's are real            
            >>> assert (A**K1.real)*(A**K2.real)==A**(K1.real+K2.real) if (
            ...     (not flat) or (K1.real>=0 and K1.real>=0)
            ... ) else True

            >>> assert A**K1.real/A**K2.real==A**(K1.real-K2.real) if (
            ...     not flat or K1.real>= 0 and K2.real <= 0
            ... ) else True
            
        Zero power has some interesting properties: 
            
            The resulting ellipse is always a unit sphere, 
            the mean is wherever it gets stretched to while we 
            transform the ellipse to a sphere
              
            >>> assert Matrix((A**0).var) == numpy.ones
            >>> assert (A**0).mean == A.mean*(A**-1).transform() if not flat else True

            if there are missing dimensions the transform is irreversable so this stops working 
            >>> assert (A**0).mean == A.mean*A.transform(-1) or flat
            
        derivation of multiplication from this is messy.just remember that 
        all Mvars on the right, in a multiply, can just be converted to matrix:
            
            >>> assert (A*B).cov == (A*B.transform()+B*A.transform()).cov/2

            >>> assert (A*B).mean == (A*B.transform()+B*A.transform()).mean/2 or flat

            >>> assert M*B==M*B.transform()
            >>> assert A**2==A*A
            >>> assert A**2==A*A.transform() or flat
        """
        if numpy.real(power)<0: 
            self=self.inflate()
        
        V=self.vectors            
        dmean=self.mean-self.mean*V.H*V        
        
        return Mvar(
            mean=self.mean*self.transform(power-1)+dmean,
            vectors=self.vectors,
            var=self.var**power,
            square=bool(numpy.imag(power)),
        )  

        
    def __mul__(self,other):        
        """
        self*other
        
        coercion notes:
            All non Mvar imputs will be converted to numpy arrays, then 
            treated as constants if zero dimensional, or matrixes otherwise.

            the resulting types for all currently supported multipications are listed below            
            
            >>> assert isinstance(A*B,Mvar)
            >>> assert isinstance(A*M,Mvar)
            >>> assert isinstance(M*A,Matrix) 
            >>> assert isinstance(A*K1,Mvar)
            >>> assert isinstance(K1*A,Mvar)

            This can be explained as: 
                When multiplying by a constant the result is always an Mvar.
                When multiplying a mix of Mvars an Matrixes the result has the 
                    sametype as the leftmost operand
            
       Whenever an mvar is found on the right of a Matrix or Mvar it is replaced by a 
            self.transform() matrix and the multiplication is re-called.
            
        general properties:
            
            Scalar multiplication fits with addition so:
                >>> assert A+A == 2*A
                >>> assert (2*A).mean==2*A.mean
                >>> assert (2*A.cov) == 2*A.cov
            
            This is different from multiplication by a scale matrix which gives
                >>> assert (A*(K1*E)).mean == K1*A.mean
                >>> assert (A*(K1*E)).cov == A.cov*abs(K1)**2

            constants still commute:          
                >>> assert K1*A*M == A*K1*M 
                >>> assert K1*A*M == A*M*K1

            constants are still asociative
                >>> assert (K1*A)*K2 == K1*(A*K2)

            so are matrixes if the Mvar is not in the middle, because it's all matrix multiply.
                >>> assert (A*M)*M2 == A*(M*M2)
                >>> assert (M*M2)*A == M*(M2*A)

            if you mix mvars with matrixes, it's two different types of multiplication, and 
            so is not asociative
                >>> (M*A)*M2 == M*(A*M2)
                False
                
            and because of that, this also doesn't work, except in 1-dimension,
                >>> (A*B)*C == A*(B*C) if ndim > 1  else False
                False

            the reason that those don't work boils down to:            
                >>> A.transform()*M == (A*M).transform()
                False

            if you work it out you'll find that the problem is unavoidable given:
                >>> assert (A*M).cov == M.H*A.cov*M
                >>> assert (A**2).transform() == A.cov

            multiplication is distributive for constants only.
                >>> assert A*(K1+K2)==A*K1+A*K2
                >>> A*(M+M2)==A*M+A*M2
                False
                >>> A*(B+C)==A*B+A*C
                False
             
                The reason is more clear if you consider the following:
                >>> A*(E+E) == A*E+A*E
                False

                because the left side will do a matrix multiplication by 2, 
                and the right will do a scalar multiplication by 2, the means will match but the cov's will not 
                
                The pure mvar case fails for slightly different reasons:
                    >>> assert A*(B**0+B**0) == A*(2*B**0)   #here the mean is stretched to sqrt(2) times 
                    >>> assert (2*B**0).transform() == sqrt(2)*(B**0).transform()    
        
                    >>> assert (A*B**0 + A*B**0).cov == (2*A*B**0).cov 
                    >>> (A*B**0 + A*B**0).mean == (2*A*B**0).mean
                    False

        for notes 
            
        given __mul__ and __pow__ it would be immoral to not overload divide as well, 
        The Automath class takes care of these details
            A/?
            
            >>> assert A/B == A*(B**(-1))
            >>> assert A/M == A*(M**(-1))
            >>> assert A/K1 == A*(K1**(-1))
        
            ?/A: see __rmul__ and __pow__
            
            >>> assert K1/A == K1*(A**(-1))
            >>> assert M/A==M*(A**(-1))

        assert Matrix((A**0.0).trace()) == A.shape[0]
        """
        other=self._mulConvert(other)
        return self._multipliers[type(other)](self,other) 
    
    def _scalarMul(self,scalar):
        """
        self*scalar

        >>> assert A*K1 == K1*A

            Matrix multiplication and scalar multiplication behave differently 
            from eachother.  
            
            For this to be a properly defined vector space scalar 
            multiplication must fit with addition, and addition here is 
            defined so it can be used in the kalman noise addition step so: 
            
            >>> assert (A+A)==(2*A)
            
            >>> assert (A+A).mean==(2*A).mean
            >>> assert (A+A).mean==2*A.mean
            
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

            >>> assert 1j*A*1j==-A
        """
        return Mvar(
            mean= scalar*self.mean,
            var = scalar*self.var,
            vectors = self.vectors,
            square = not numpy.isreal(scalar),
        )

    def _matrixMul(self,matrix):
        """
        self*matrix
        
            matrix multiplication transforms the mean and ellipse of the 
            distribution. Defined this way to work with the kalman state 
            update step.

            >>> assert A*E==A
            >>> assert (-A)*E==-A 
            
            simple scale is like this:
            >>> assert (A*(E*K1)).mean==A.mean*K1
            >>> assert (A*(E*K1)).cov ==(E*K1).H*A.cov*(E*K1)
            
            or with a more general transform()
            >>> assert (A*M).cov==M.H*A.cov*M
            >>> assert (A*M).mean==A.mean*M

        """
        return Mvar(
            mean=self.mean*matrix,
            var=self.var,
            vectors=self.vectors*matrix,
        )

    def _mvarMul(self,mvar):
        """
        self*mvar

        multiplying two Mvars together is defined to fit with power
        
        >>> assert A*A==A**2
        >>> assert A*A==A*A.transform() or flat
        >>> assert A*B == B*A

        Note that the result does not depend on the mean of the 
        second mvar(!) (really any mvar after the leftmost mvar or matrix)

        
        """
        result = (self*mvar.transform()+mvar*self.transform())
        
        result.mean += (
            self.mean-self.mean*mvar.transform(0)+
            mvar.mean-mvar.mean*self.transform(0)
        )

        return result/2

    @staticmethod
    def _mulConvert(
        item,
        helper=lambda item: Matrix(item) if item.ndim else item
    ):
        return (
            item if 
            isinstance(item,Mvar) else 
            helper(numpy.array(item))
        )

    def __rmul__(
        self,
        other,
    ):
        """
        other*self
        
        multiplication order doesn't matter for constants
        
            >>> assert K1*A == A*K1
        
            but it matters a lot for Matrix/Mvar multiplication
        
            >>> assert isinstance(A*M,Mvar)
            >>> assert isinstance(M*A,Matrix)
        
        be careful with right multiplying:
            Because power must fit with multiplication
        
            it was designed to satisfy
            >>> assert A*A==A**2
        
            The most obvious way to treat right multiplication by a matrix is 
            to do exactly the same thing we're dong in Mvar*Mvar, which is 
            convert the right Mvar to the square root of its covariance matrix
            and continue normally,this yields a matrix, not an Mvar.
            
            this conversion is not applied when multiplied by a constant.
        
        martix*Mvar
            >>> assert M*A==M*A.transform()

        Mvar*constant==constant*Mvar
            >>> assert A*K1 == K1*A
        
        A/?
        
        see __mul__ and __pow__
        it would be immoral to overload power and multiply but not divide 
            >>> assert A/B == A*(B**(-1))
            >>> assert A/M == A*(M**(-1))
            >>> assert A/K1 == A*(K1**(-1))

        ?/A
        
        see __rmul__ and __pow__
            >>> assert K1/A == K1*(A**(-1))
            >>> assert M/A==M*(A**(-1))
        """
        (transform,other)= self._rmulConvert(other)
        return self._rmultipliers[type(other)](transform,other)

    def _rmulConvert(self,other,
        helper=lambda self,other:(
            self.transform() if other.ndim else self,
            Matrix(other) if other.ndim else other,
        )
    ):
        return helper(self,numpy.array(other))

    _rmultipliers={
        #if the left operand is a matrix, the mvar has been converted to
        #to a matrix -> use matrix multiply
        (Matrix):lambda self,other:other*self,
        #if the left operand is a constant use scalar multiply
        (numpy.ndarray):_scalarMul
    }


    
    def __add__(self,other):
        """
        self+other
        
        Implementation:
            >>> assert (A+B)==Mvar(
            ...     mean=A.mean+B.mean,
            ...     vectors=numpy.vstack([A.vectors,B.vectors]),
            ...     var = numpy.concatenate([A.var,B.var]),
            ... )

        When using addition keep in mind that rand()+rand() is not like scaling 
        one random number by 2 (2*rand()), it adds together two random numbers.

        The add here is like rand()+rand()
        
        Addition is defined this way so it can be used directly in the kalman 
        noise addition step
        
        so if you want simple scale use matrix multiplication like rand()*(2*eye)
        
        scalar multiplication however fits with addition:

        >>> assert (A+A).mean==(2*A).mean
        >>> assert (A+A).mean==2*A.mean
        
        >>> assert (A+B).mean==A.mean+B.mean
        >>> assert (A+B).cov==A.cov+B.cov
        
        watch out subtraction is the inverse of addition 
            >>> assert A-A == Mvar(mean=numpy.zeros_like(A.mean))
            >>> assert (A-B)+B == A
            >>> assert (A-B).mean == A.mean - B.mean
            >>> assert (A-B).cov== A.cov - B.cov
            
        if you want something that acts like rand()-rand() use an eye to scale:
            
            >>> assert (A+B*(-E)).mean == A.mean - B.mean
            >>> assert (A+B*(-E)).cov== A.cov + B.cov

        __sub__ should also fit with __neg__, __add__, and scalar multiplication.
        
            >>> assert B+(-A) == B+(-1)*A == B-A
            >>> assert A-B == -(B-A)
            >>> assert A+(B-B)==A
            
            but watchout you'll end up with complex... everything?
        """
        #todo: fix the crash generated, for flat objects by: 1/A-1/A (inf-inf == nan)

        other = other if isinstance(other,Mvar) else Mvar(mean=other)
        return Mvar(
            mean=self.mean+other.mean,
            vectors=numpy.vstack([self.vectors,other.vectors]),
            var = numpy.concatenate([self.var,other.var]),
        )
        
    ################# Non-Math python internals
    def __call__(self,locations):
         """
        self(locations)

         Returns the probability density in the specified locations, 
         The vectors should be aligned onto the last dimension
         That last dimension is squeezed out during the calculation
 
         If spacial dimensions have been flattened out of the mvar the result is always 1/0
         since the probablilities will have dimensions of hits/length**ndim 
         """
         return numpy.exp(self.dist2(self,locations))/2/numpy.pi/sqrt(self.det(self))
 
    def __repr__(self):
        """
        print self
        """
        return '\n'.join([
            'Mvar(',
            '    mean=',8*' '+self.mean.__repr__().replace('\n','\n'+8*' ')+',',
            '    var=',8*' '+self.var.__repr__().replace('\n','\n'+8*' ')+',',
            '    vectors=',8*' '+self.vectors.__repr__().replace('\n','\n'+8*' ')+',',
            ')',
        ])
        
    
    __str__=__repr__

    ################ Art
    
    def patch(self,nstd=2,alpha='auto',slope=1,minalpha=0.05,**kwargs):
        """
            get a matplotlib Ellipse patch representing the Mvar, 
            all **kwargs are passed on to the call to 
            matplotlib.patches.Ellipse

            not surprisingly Ellipse only works for 2d data.

            the number of standard deviations, 'nstd', is just a multiplier for 
            the eigen values. So the standard deviations are projected, if you 
            want volumetric standard deviations I think you need to multiply by sqrt(ndim)

            if  you don't specify a value for alpha it is set to the exponential of the area,
            as if it has a fixed amount if ink that is spread over it's area.

            the 'slope' and 'minalpha' parameters control this auto-alpha:
                'slope' controls how quickly the the alpha drops to zero
                'minalpha' is used to make sure that very large elipses are not invisible.  
        """
        if self.ndim != 2:
            raise ValueError(
                'this method can only produce patches for 2d data'
            )
        
        if alpha=='auto':
            kwargs['alpha']=numpy.max([
                minalpha,
                numpy.exp(-slope*sqrt(self.det()))
            ])
        else:
            kwargs['alpha']=alpha

        #unpack the width and height from the scale matrix 
        wh = nstd*sqrt(self.var)
        wh[~numpy.isfinite(wh)]=10**5

        width,height=2*wh

        #return an Ellipse patch
        return Ellipse(
            #with the Mvar's mean at the centre 
            xy=tuple(self.mean.flatten()),
            #matching width and height
            width=width, height=height,
            #and rotation angle pulled from the vectors matrix
            angle=180/numpy.pi*(numpy.angle(helpers.ascomplex(self.vectors)[0,0])),
            #while transmitting any kwargs.
            **kwargs
        )

Mvar._multipliers={
    Mvar:Mvar._mvarMul,
    Matrix:Mvar._matrixMul,
    numpy.ndarray:Mvar._scalarMul
}

## extras    

def wiki(P,M):
    """
    Direct implementation of the wikipedia blending algorithm
    
    The quickest way to prove it's equivalent is by examining these:

    >>> if not flat:    
    ...     assert A**-1 == A*A**-2 or flat
    ...     assert A & B == (A*A**-2+B*B**-2)**-1 or flat
    ...
    ...     D = A*(A.cov)**(-1) + B*(B.cov)**(-1)
    ...     assert wiki(A,B) == D*(D.cov)**(-1)
    ...     assert A & B == wiki(A,B)
    """
    yk=M.mean-P.mean
    Sk=P.cov+M.cov
    Kk=P.cov*Sk.I
    
    return Mvar.fromCov(
        mean=(P.mean + yk*Kk.H),
        cov=(Matrix.eye(P.ndim)-Kk)*P.cov
    )

def mooreGiven(self,index,value):
    """
    direct implementation of the "given" algorithm in
    Andrew moore's data-mining/gussian slides

    todo: figure out why this doesn't work
 
    >>> assert mooreGiven(A,index=0,value=1)==A.given(index=0,value=1)
    """
    Iv=binindex(index,self.ndim)
    Iu=~Iv
 
    U=self[Iu]
    V=self[Iv]

    vu=numpy.diagflat(self.var)*V.vectors.H*U.vectors

    return Mvar.fromCov(
        mean=U.mean+(value-V.mean)*(V.transform(power=-2))*vu,
        cov=U.cov-vu.H*(V.transform(power=-2))*vu,
    )

def binindex(index,n):
    """
    convert whatever format index, for this object, to binary 
    so it can be easily inverted
    """
    if hasattr(index,'dtype') and index.dtype==bool:
        return index
    
    binindex=numpy.zeros(n,dtype=bool)
    binindex[index]=True

    return binindex


if __name__=='__main__':    
    #overwrite everything we just created with the copy that was 
    #created when we imported mvar; there can only be one.
    from mvar import *
    from testObjects import *
    
    mooreGiven(A,0,1)==A.given(0,1)

    AB= Mvar.stack(A,B)
    AB[:A.ndim]==A
    AB[A.ndim:]==B
