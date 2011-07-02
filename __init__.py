#! /usr/bin/env python

#todo: formData should be the main constructor
#todo: the approximation tollerence should be a class attribute
#todo: merge chain and  sensor.measure??
#todo: type handling- be more uniform, arrays work element wize, matrixes get converted to Mvars ?
#todo: wiki: Complex Normal Distribution (I knew there was something underconstrained about these)
#todo: Mvar.real & Mvar.imag?,  
#todo: consider removing the autosquare if you ever want to speed things up, 
#         I bet it would help. it would also allow other factorizations (like cholsky) 
#todo: wikipedia/kalmanfiltering#information filter    !!   the mvar and it's 
#       inverse are different things, maybe the linear algebra should go in a 
#       super class, and all the covariance/information filter stuff in two sub classes 
#todo: better type handling, multimethods? many things that accept an mvar should 
#        accept Mvar.eye, Mvar.zeros, Mvar.infs 
#todo: error handling
#todo: do something about mvars with zero dimensions ?
#todo: understand transforms composed of Mvars as the component vectors, and 
#          whether it is meaningful to consider mvars in both the rows and columns
#todo: implement a transpose,for the above 
#todo: chi*2 distribution for the lengths (other distributions)
#todo: see if div should mtlab like matlab backwards divide added
#todo: impliment collections so that '|' is meaningful
#todo: cleanup my 'square' function (now that it is clear that it's an SVD)
#todo: implement transpose and dot product, related to bilinear forms?
#todo: split the class into two levels: "fast" and 'safe'? <- "if __debug__" ?
#      maybe have the 'safe' class inherit from 'fast' and a add a variance-free 'plane' class?
#todo: understand the EM and K-means algorithms (available in scipy)
#todo: understans what complex numbers imply with these.
#todo: understand the relationship between these and a hessian matrix.
#todo: figure out the relationship between these and spherical harmonics
#todo: investigate higher order cumulants, 'principal cumulant analysis??'

"""
This module contains one thing: the "Mvar" class.

Mvar is the main idea of the module: Multivariate normal distributions 
    packaged to act like a vector. Perfect for kalman filtering, sensor fusion, 
    (and maybe Expectation Maximization)  

there are also a couple of loose functions like 'wiki', which is just to demonstrate 
the equivalency between my blending algorithm, and the wikipedia version of it.
        http://en.wikipedia.org/wiki/Kalman_filtering#Update

The docstrings are full of examples. The objects used in the examples are created 
by test.sh, and stored in test_objects.pkl. You can get the most recent versions of them by 
importing testObjects.py, which will give you a module containing the objects used

in the doc examples
    A,B and C are instances of the Mvar class  
    K1 and K2 are random numbers
    M and M2 are matrixes
    E is an apropriately sized eye matrix
    N is an integer

remember: circular logic works because circluar logic works.
    a lot of the examples are demonstrations of what the code is doing, or expected
    invariants. They don't prove I'm right, but only that I'm being consistant
 
"""
############  imports

## 3rd party
import numpy

## local
import helpers
from helpers import sqrt
from square import square
from matrix import Matrix

#decorations
import decorate

#base class
from plane import Plane

Mvar=decorate.underConstruction('Mvar')
Mvar.T=decorate.underConstruction('Mvar.T')

def format(something):
    '''
    take an arraylike object and return a Matrix, array, or the object 
    (unmodified), depending on the dimensionality of the input 
    '''
    A=numpy.array(something)
                 
    if A.ndim == 2:
        something=numpy.asmatrix(A)
        something.__class__=Matrix
    elif A.ndim != 0:
        something=A
    
    return something


@decorate.prepare(lambda data,mean:[format(data),format(mean)])
@decorate.MultiMethod
def fromData(data,mean=None,**kwargs):
    """
    optional arguments:
        mean - array like, (1,ndim) 
        weights - array like, (N,) 
        bias - bool

        anything else is passed through to from cov

    >>> assert Mvar.fromData(A)==A 
    
    >>> data=[1,2,3]
    >>> new=Mvar.fromData(data)
    >>> assert new.mean == data
    >>> assert Matrix(new.var) == Matrix.zeros
    >>> assert new.vectors == Matrix.zeros
    >>> assert new.cov == Matrix.zeros
    
    bias is passed to numpy's cov function.
    
    any kwargs are just passed on the Mvar constructor.
    
    this creates an Mvar with the same mean and covariance as the supplied 
    data with each row being a sample and each column being a dimenson
    
    remember numpy's default covariance calculation divides by (n-1) not 
    (n) set bias = false to use n-1,
    """
    return fromMatrix(Matrix(data),**kwargs)

@fromData.register(Mvar,type(None))
def fromMvar(self,mean=None):
    """
    >>> assert fromData(A)==A
    """
    return self.copy(deep = True)

@fromData.register(Mvar)
def fromMvarOffset(self,mean=Matrix.zeros):
    """
    think paralell axis theorem
    
    >>> a=A[0]
    >>> assert fromData(a,mean=0).mean == Matrix.zeros
    >>> assert fromData(a,mean=0).cov == a.cov+a.mean.H*a.mean

    >>> assert fromData(A,mean=Matrix.zeros).mean == Matrix.zeros
    >>> assert fromData(A,mean=Matrix.zeros).cov == A.cov+A.mean.H*A.mean
    """
    if callable(mean):
        mean=mean(self.ndim)

    delta=(self-mean)

    vectors = delta.mean

    subVectors=delta.vectors 
    subWeights=delta.var
     
    return Mvar(
        mean = mean,
        var = numpy.concatenate([[1],subWeights]),
        vectors = numpy.concatenate([vectors,subVectors]),
    )

@fromData.register(numpy.ndarray)
def fromArray(data,mean=None,weights=None,bias=True):
    """
    >>> data1 = numpy.random.randn(100,2)+5*numpy.random.randn(1,2)
    >>> data2 = numpy.random.randn(100,2)+5*numpy.random.randn(1,2)
    >>>
    >>> mvar1 = Mvar.fromData(data1)
    >>> mvar2 = Mvar.fromData(data2)
    >>>
    >>> assert Mvar.fromData([mvar1,mvar2]) == Mvar.fromData(numpy.vstack([data1,data2]))

    >>> N1=1000
    >>> N2=10
    >>> data1 = numpy.random.randn(N1,2)+5*numpy.random.randn(1,2)
    >>> data2 = numpy.random.randn(N2,2)+5*numpy.random.randn(1,2)
    >>>
    >>> mvar1 = Mvar.fromData(data1)
    >>> mvar2 = Mvar.fromData(data2)
    >>>
    >>> assert Mvar.fromData([mvar1,mvar2],weights=[N1,N2]) == Mvar.fromData(numpy.vstack([data1,data2]))
    """
    if data.dtype is not numpy.dtype('object'):
        return fromMatrix(Matrix(data),weights=weights,mean=mean,bias=bias)

    ismvar=numpy.array([isinstance(vector,Mvar) for vector in data])    
    mvars=data[ismvar]

    data=numpy.array([
        numpy.squeeze(vector.mean if mvar else vector)
        for mvar,vector 
        in zip(ismvar,data)
    ])

    N=getN(data,weights)-(not bias)
    weights=getWeights(weights,data,N)
    mean = getMean(data,mean,weights)

    subVectors=numpy.vstack([
        mvar.vectors 
        for mvar in mvars
    ])

    subWeights=numpy.concatenate([
        w*mvar.var
        for w,mvar in zip(weights[ismvar],mvars)
    ])

    vectors=data-numpy.array(mean)

    return Mvar(
        mean = mean,
        var = numpy.concatenate([weights,subWeights]),
        vectors = numpy.concatenate([vectors,subVectors]),
    )

def getN(data,weights):
    return (
        data.shape[0] 
        if weights is None 
        else numpy.sum(weights)
    )


def getWeights(weights,data,N):
    return (
        numpy.ones(data.shape[0])
        if weights is None 
        else numpy.array(weights)
    )/float(N)

def getMean(data,mean,weights):
    if mean is None:
        mean = numpy.multiply(weights[:,None],data).sum(0)
    elif callable(mean):
        mean = mean(data.shape[1])

    mean = numpy.asmatrix(mean)

    return mean



@fromData.register(Matrix)
def fromMatrix(data,mean=None,weights=None,bias=True,**kwargs):
    """
    >>> D=Mvar.fromData([[0],[2]])
    >>> assert D.mean == 1
    >>> assert D.var == 1

    >>> D=Mvar.fromData([[0],[2]],mean=[0])
    >>> assert D.mean == 0
    >>> assert D.var == 2

    """
    N = getN(data,weights)-(not bias)
    weights = getWeights(weights,data,N)
    mean = getMean(data,mean,weights)

    vectors=data-mean

    return Mvar(
        mean=mean,
        var=weights,
        vectors=vectors,
    )


@decorate.MultiMethod.sign(Mvar)
class Mvar(Plane):
    """
    Multivariate normal distributions packaged to act like a vector 
    (Ref: andrew moore / data mining / gaussians )
    (Ref: http://en.wikipedia.org/wiki/Vector_space)
    
    The class fully supports complex numbers.
    
    basic math operators (+,-,*,/,**,&) have been overloaded to work 'normally'
    But there are several surprising features in the math these things produce,
    so watchout. 
    
    designed for kalman filtering, sensor fusion, or maybe expectation 
    maximization and principal component analysis 
    (ref: http://en.wikipedia.org/wiki/Expectation-maximization_algorithm)

    kalman filtering: state[t+1] = (state[t]*STM + noise) & measurment
    Sensor fusion:    result = measurment1 & measurrment2 & measurment3
        
    
    Attributes:
        mean: mean of the distribution
        var:  the variance asociated with each vector
        vectors: unit eigen-vectors, as rows
        
    Properties:
        ndim: the number of dimensions of the space we're working in
        cov : get or set the covariance matrix    
        scaled: get the vectors, scaled by one standard deviation (transforms from unit-eigen-space to data-space) 
      
    No work has been done to make things fast, because there is no point until they work at all 
    """
    fromData=staticmethod(fromData)    

    ############## Creation
    def __init__(self,
        vectors=Matrix.eye,
        var=numpy.ones,
        mean=numpy.zeros,
        square=True,
        squeeze=True,
    ):
        """
        Create an Mvar from available attributes.
        
        vectors: defaults to zeros
        var: (variance) defaults to ones
        mean: defaults to zeros
        
        >>> assert A.var*numpy.array(A.vectors.H)*A.vectors == A.cov
                
        set 'square' to false if you know your vectors already form a unitary matrix. 
        set 'squeeze' to false if you don't want small variances, <1e-12, to  automatically removed
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
            self.copy(self.squeeze())

    ############## alternate creation methods
    @staticmethod
    def fromCov(cov,**kwargs):
        """
        everything in kwargs is passed directly to the constructor
        """
        cov=Matrix(cov)

        diag = Matrix(numpy.diag(cov))
        eig = numpy.linalg.eigh if abs(diag) == diag else numpy.linalg.eig
        #get the variances and vectors.
        (var,vectors) = eig(cov) if cov.size else (Matrix.zeros([0,1]),Matrix.zeros([0,0]))
        vectors=Matrix(vectors.H)     

        return Mvar(
            vectors=vectors,
            var=var,
            square=False,
            **kwargs
        )

    @staticmethod
    def zeros(n=1,mean=None):
        """
        >>> n=abs(N)
        >>> Z=Mvar.zeros(n)
        >>> assert Z.mean==Matrix.zeros
        >>> assert Z.var.size==0
        >>> assert Z.vectors.size==0
        >>> assert Z**-1 == Mvar.infs
        """
        return Mvar(mean=Matrix.zeros(n) if mean is None else Mean)
    
    @staticmethod
    def infs(n=1,mean=None):
        """
        >>> n=abs(N)
        >>> inf=Mvar.infs(n)
        >>> assert inf.mean==Matrix.zeros
        >>> assert inf.var.size==inf.mean.size==n
        >>> assert Matrix(inf.var)==Matrix.infs
        >>> assert inf.vectors==Matrix.eye
        >>> assert inf**-1 == Mvar.zeros
        """
        result = Mvar.zeros(n)**-1
        if mean is not None:
            result.mean = mean
        return result

    @staticmethod
    def eye(n=1,mean = None):
        """
        >>> n=abs(N)
        >>> eye=Mvar.eye(n)
        >>> assert eye.mean==Matrix.zeros
        >>> assert eye.var.size==eye.mean.size==n
        >>> assert Matrix(eye.var)==Matrix.ones
        >>> assert eye.vectors==Matrix.eye
        >>> assert eye**-1 == eye
        """
        return Mvar(
            mean=Matrix.zeros([1,n]) if mean is None else mean,
            vectors=Matrix.eye(n),
        )
    
    ##### 'cosmetic' manipulations
    def inflate(self):
        """
        add the zero length direction vectors so no information is lost when using 
        the vectors parameter

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
        drop any vector/variance pairs with (self.var) under 1e-12,
        """
        result=self.copy()
        
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
        result=self.copy()
        (result.var,result.vectors)=square(
            vectors=self.vectors,
            var=self.var,
        )
        return result

    
    ############ setters/getters -> properties
    @decorate.prop
    class cov():
        """
        get or set the covariance matrix used by the object
        
        >>> assert A.cov==A.var*numpy.array(A.vectors.H)*A.vectors
        >>> assert abs(A).cov==A.scaled.H*A.scaled
        """
        def fget(self):
            return self.var*numpy.array(self.vectors.H)*self.vectors

        def fset(self,cov):
            new=Mvar.fromCov(
                mean=self.mean,
                cov=cov,
            )
            self.copy(new)
    

    @decorate.prop
    class scaled():
        """
        get the vectors, scaled by the standard deviations. 
        Useful for transforming from unit-eigen-space, to data-space

        >>> assert A.vectors.H*A.scaled==A.transform()
        """
        def fget(self):
            return Matrix(sqrt(self.var[:,None])*numpy.array(self.vectors))
        
    
    @decorate.prop
    class flat():
        """
        >>> assert bool(A.flat) == bool(A.vectors.shape[1] > A.vectors.shape[0]) 
        """
        def fget(self):
            return max(self.vectors.shape[1] - self.vectors.shape[0],0)

    @decorate.prop
    class ndim():
        """
        get the number of dimensions of the space the mvar exists in
        >>> assert A.ndim==A.mean.size
        """
        def fget(self):
            return self.mean.size

    @decorate.prop
    class rank():
        """
        get the number of dimensions of the space covered by the mvar
        >>> assert A.rank == A.var.size
        """
        def fget(self):
            return self.mean.size

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
        >>> assert (A**numpy.real(K1)).transform() == A.transform(numpy.real(K1)) 

        >>> assert (A*B.transform() + B*A.transform()).cov/2 == (A*B).cov

        >>> assert Matrix(numpy.trace(A.transform(0))) == A.shape[0] 
        """
        if not self.var.size:
            ndim=self.ndim
            shape=(ndim,ndim)
            return Matrix.zeros(shape)

        if not numpy.isreal(self.var).all() or not(self.var>0).all():
            power = complex(power)


        if helpers.approx(power):
            vectors=self.vectors
            varP=numpy.ones_like(self.var)
        else:
            #the null vectors are automatically being ignored,
            #ignore the infinite ones as well
            keep=~helpers.approx(self.var**(-1))
            
            varP=numpy.real_if_close(self.var[keep]**(power/2.0))
            vectors=self.vectors[keep,:]

        return varP*numpy.array(vectors.H)*vectors

    def sign(self):
        return helpers.sign(self.var)

    ########## Utilities
        
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
        If you're trying to do that use a better matrix multiply, or Mvar.chain 
        
        see also Mvar.chain
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
        #todo look at that complex-normal distributions
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

        >>> assert A.det() == (
        ...     0 if 
        ...     A.shape[0]!=A.shape[1] else 
        ...     numpy.prod(A.var)
        ... )

        if you want the pseudo-det use self.var.prod()
        (ref: http://en.wikipedia.org/wiki/Pseudo-determinant)
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

        >>> assert Matrix((A+B).trace()) == A.trace()+B.trace()
        
        >>> assert Matrix((A*B).trace()) == (B*A).trace()
        >>> assert Matrix((A.cov*B.cov).trace()) == (A*B).trace() 
        >>> assert Matrix((A*B.transform()).trace()) == (B*A.transform()).trace() 
        """
        return self.var.sum()
    
    def width(self):
        """
        return the standard deviations of the mvar, along each coordinate-axis.
        (not eigen-axes).
        
        >>> assert Matrix([A[n].var[0] for n in range(A.ndim)]) == A.width()**2

        >>> norm = A/A.width()
        >>> corr = norm.cov
        >>> assert Matrix(corr.diagonal()) == Matrix.ones
        >>> assert Matrix([norm[n].var[0] for n in range(norm.ndim)]) == Matrix.ones

        This is very different from 

        >>> assert Matrix((A**0).var) == Matrix.ones

        because doing it with power scales along the eigenvectrs, this scales along the axes
        """
        S=numpy.array(self.scaled)
        return (S.conj()*S).sum(0)**(0.5)


    #todo: merge with sensor.measure
    def chain(self,sensor=None,transform=None):
        """
        given a distribution of actual values and an Mvar to act as a sensor 
        this method returns the joint distribution of actual and measured values

        self:       is the value we're  taking a measurment of

        transform:  specifies the transform from the actual value to the sensor output
                    defalults to Matrix.eye

        sensor:     specifies the sensor's bias and noise
                    defaults to Mvar.zeros

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
        >>> a=Mvar.fromData(dataA)
        >>> assert a.chain()==Mvar.fromData(numpy.hstack([dataA,dataA]))        
        >>> assert a.chain(transform=M) == Mvar.fromData(dataA*numpy.hstack([E,M]))
        >>> assert a.chain(transform=M) == Mvar.fromData(numpy.hstack([dataA,dataA*M]))
        
        >>> assert a.chain(B*M,M) == a.chain(transform=M)+Mvar.stack(Mvar.zeros(a.ndim),B*M)

        reference: andrew moore/data mining/gaussians
        """
        twice = Mvar(
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
        sensor = sensor if sensor is not None else Mvar.zeros(transform.shape[1])
        #add some leading seros so it fits
        sensor=Mvar.zeros(self.ndim).stack(sensor)

        return perfect+sensor

    def dist2(self,locations):
        """
        return the square of the mahalabois distance from the Mvar to each vector.
        the vectors should be along the last dimension of the array.
   
        >>> N=1000
        ... Z=3
        ... dist = A.dist2(A.sample(N))
        ... assert (dist.mean()-A.ndm)**2 < (Z**2)*(dist.var()/N)

        
        """
        #make sure the mean is a flat numpy array
        mean=numpy.array(self.mean).squeeze()
        #and subtract it from the locations (vectors aligned to the last dimension)
        deltas=numpy.array(locations)-mean
        
        scaled=numpy.array(numpy.inner(deltas,(self**-1).scaled))
        return (scaled*scaled.conjugate()).sum(axis=locations.ndim-1)

        
    ############## indexing
    
    def given(self,index,value):
        """
        return an mvar representing the conditional probability distribution, 
        given the values, on the given indexes
        
        equivalent to: andrew moore/data mining/gussians/page 22
        except that my __and__ handels infinities, other versions of given don't
        
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
        fixed=binindex(index,self.ndim)
        if fixed.all():
            return Mvar.fromData(value)

        free = ~fixed

        Z=numpy.zeros
        meanType=(Z([],self.mean.dtype)+Z([],value.mean.dtype)).dtype
        varType=(Z([],self.var.dtype)+Z([],value.var.dtype)).dtype
        vectorType=(Z([],self.vectors.dtype)+Z([],value.vectors.dtype)).dtype

        #create the mean, for the new object,and set the values of interest
        mean=numpy.zeros([1,self.shape[0]],dtype=meanType)
        mean[:,fixed]=value.mean

        #create empty vectors for the new object
        vectors=numpy.zeros([
            value.shape[0]+(self.ndim-value.ndim),
            self.ndim,
        ],dtype=vectorType)
        vectors[0:value.shape[0],fixed]=value.vectors
        vectors[value.shape[0]:,free]=numpy.eye(self.ndim-value.ndim)
        
        #create the variance for the new object
        var=numpy.zeros(vectors.shape[0],dtype=varType)
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

        """
        self.copy(self.given(index,value))

    def __getitem__(self,index):
        """
        self[index]
        return the marginal distribution,
        over the indexed dimensions,
        """
        index = numpy.asarray(index) if hasattr(index,'__iter__') else index
        #todo: consider also indexing by eigenvalue
        #only makes sense if you have a self.sort, or self.sorted
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

        when the comparison is done with a function, the function is called 
        with the size of the self as the only argument:
            >>> n=abs(N)
            >>> assert Mvar.zeros(n) == Mvar.zeros
            >>> assert Mvar.eye(n) == Mvar.eye
            >>> assert Mvar.infs(n) == Mvar.infs
        """
        if not isinstance(other,Mvar):
            if callable(other):
                other = other(self.ndim)

        other=Mvar.fromData(other)
        
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
   
        H=lambda M:M.H*M

        SFvectors=self.vectors[Sfinite]
        SFvar=self.var[Sfinite]

        OFvectors = other.vectors[Ofinite]
        OFvar = other.var[Ofinite]

        cov=lambda vectors,var: var*numpy.array(vectors.H)*vectors

        #compare the finite and infinite covariances 
        return (
            cov(SFvectors,SFvar) == cov(OFvectors,SFvar) and
            SIvectors.H*SIvectors == OIvectors.H*OIvectors
        )

    def __gt__(self,lower):
        """
        >>> assert (A > Matrix.infs(A.ndim)) == 0
        >>> assert (A > -Matrix.infs(A.ndim)) == 1
        
        >>> AV = A*A.vectors.H
        >>> assert Matrix(AV>AV.mean) == 2**-AV.ndim 

        see doc for Mvar.inbox
        """
        return self.inBox(
            lower,
            numpy.inf*numpy.ones(self.mean.size)
        )
        
    def __ge__(self,lower):
        """
        see doc for Mvar.inbox
        """
        return self>lower

    def __lt__(self,upper):
        """
        >>> assert (A < Matrix.infs(A.ndim)) == 1
        >>> assert (A < -Matrix.infs(A.ndim)) == 0
        
        >>> AV = A*A.vectors.H
        >>> assert Matrix(AV<AV.mean) == 2**-AV.ndim 

        see doc for Mvar.inbox
        """
        return self.inBox(
            -numpy.inf*numpy.ones(self.mean.size),
            upper,
        )

    def __le__(self,upper):
        """
        see doc for Mvar.inbox
        """
        return self<upper

    def inBox(self,lower,upper):
        """
        returns the probability that all components of a sampe are between the 
        lower and upper limits 

        todo: this could (should) be expanded to return a gaussian mixture, with one (Mvar) component instead of just a  weight...
        """
        #todo: vectorize?
        lower=lower-self.mean
        upper=upper-self.mean

        if isinstance(lower,Mvar):
            l=lower.mean
            lower.mean=Matrix.zeros
            self = self+lower
            lower=l
        if isinstance(upper,Mvar):
            u=upper.mean
            upper.mean=Matrix.zeros
            self = self+upper
            upper=u

        lower=numpy.array(lower).flatten()
        upper=numpy.array(upper).flatten()

        if self.ndim == 1:
            self=norm(0,self.var**0.5)
            return self.cdf(upper)-self.cdf(lower)

        Iwidth=self.width()**-1
        
        self = self*Iwidth
        lower = lower*Iwidth
        upper = upper*Iwidth       

        
        try: 
            mvs = mvstdnormcdf
        except NameError:
            from mvncdf import mvstdnormcdf as mvs
            globals().update({'mvstdnormcdf':mvs})

        return mvs(lower,upper,self.cov)
        
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

        >>> assert (A & ~A) == Mvar(mean=A.mean, vectors=A.vectors, var=Matrix.infs)
        >>> if not A.flat:
        ...     assert (A & ~A) == Mvar(mean=numpy.zeros(A.ndim))**-1


        infinite variances provide no information, having a no effect when blended
        >>> if not B.flat:
        ...     assert A == A & (B & ~B)
        
        if the mvar is flat, things are a little different:
            like this you're taking a slice of A in the plane of B
            >>> assert  A &(B & ~B) == A & Mvar(mean=B.mean, vectors=B.vectors, var=Matrix.infs)
   
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
        return self+other-(self&other)

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

        >>> if not (A.flat or B.flat): 
        ...     assert A & B == 1/(1/A+1/B)

        >>> if not (A.flat or B.flat or C.flat):
        ...     import operator
        ...     abc=numpy.random.permutation([A,B,C])
        ...     assert A & B & C == helpers.paralell(*abc)
        ...     assert A & B & C == reduce(operator.and_ ,abc)
        ...     assert (A & B) & C == A & (B & C)

        >>> if not A.flat:
        ...     assert A &-A == Mvar(mean=numpy.zeros(ndim))**-1
        ...     assert A &~A == Mvar(mean=numpy.zeros(ndim))**-1


        >>> assert (A & A).cov == A.cov/2
        >>> assert (A & A).mean == A.mean
                
        The proof that this is identical to the wikipedia definition of blend 
        is a little too involved to write here. Just try it (and see the "wiki"
        function)
        
        >>> if not (A.flat or B.flat):
        ...     assert A & B == wiki(A,B)

        this algorithm is also, at the same time, solving linear equations
        where the zero vatiances correspond to a plane's null vectors 

        >>> L1=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf)
        >>> L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[1,1]
        >>> assert (L1&L2).var.size==0

        >>> L1=Mvar(mean=[1,0],vectors=[1,1],var=numpy.inf)
        >>> L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[2,1]
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
        
        Dmean = Plane.__and__(self,other).mean

        #do the blending, while compensating for the mean of the working plane
        return ((self-Dmean)**-1+(other-Dmean)**-1)**-1+Dmean

    def __pow__(self,power):
        """
        self**power

        >>> #the transform version doesn't work for flat objects if the transform power is less than 0
        >>> k = numpy.real(K1)
        >>> if not A.flat or k>0:
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
        all Mvars on the right, in a multiply, can just be converted to matrix:
            
            >>> assert (A*B).cov == (A*B.transform()+B*A.transform()).cov/2

            >>> assert (A*B).mean == (A*B.transform()+B*A.transform()).mean/2 or (A.flat or B.flat)

            >>> assert M.H*B==M.H*B.transform()
            >>> assert A**2==A*A
            >>> if not A.flat:
            ...     #information is lost if the object is flat.
            ...     assert A**2==A*A.transform()
        """
        if numpy.real(power)<0: 
            self=self.inflate()
        
        V=self.vectors            
        dmean=self.mean-self.mean*V.H*V        
        
        
        return Mvar(
            mean=self.mean*self.transform(power-1)+dmean,
            vectors=self.vectors,
            var=(
                self.var**power
                #Interesting idea but wrong: it would only fix A*A==A**2 for complex objects.  
                #if numpy.isreal(self.var).all() else 
                #numpy.conj(self.var**((power-1)/2))*self.var*self.var**((power-1)/2)
            ),
            square=False#bool(numpy.imag(power)),
        )

    @decorate.prepare(lambda self,other:(self,format(other)))
    @decorate.MultiMethod
    def __mul__(self,other):        
        """
        self*other
        
        coercion notes:
            All non Mvar imputs will be converted to numpy arrays, then 
            treated as constants if zero dimensional, or matrixes otherwise.

            the resulting types for all currently supported multipications are listed below            
            
            >>> assert isinstance(A*B,Mvar)
            >>> assert isinstance(A*M,Mvar)
            >>> assert isinstance(M.T*A,Matrix) 
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

            1d arrays in a multipliction are interpreted as the diagonal in 
            a matrix multiply
                >>> assert (A*(K1*E)).mean == (A*[K1]).mean
                >>> assert (A*(K1*E)).cov == (A*[K1]).cov

            constants still commute:          
                >>> assert K1*A*M == A*K1*M 
                >>> assert K1*A*M == A*M*K1

            constants are still asociative
                >>> assert (K1*A)*K2 == K1*(A*K2)

            so are matrixes if the Mvar is not in the middle, because it's all matrix multiply.
                >>> assert (A*M)*M2.H == A*(M*M2.H)
                >>> assert (M*M2.H)*A == M*(M2.H*A)

            if you mix mvars with matrixes, it's two different types of multiplication, and 
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
                
                The pure mvar case fails for slightly different reasons:
                    >>> assert A*(B**0+B**0) == A*(2*B**0)   #here the mean is stretched to sqrt(2) times 
                    >>> assert (2*B**0).transform() == sqrt(2)*(B**0).transform()    
        
                    >>> assert (A*B**0 + A*B**0).cov == (2*A*B**0).cov 
                    >>> assert (A*B**0 + A*B**0).mean != (2*A*B**0).mean

        for notes 
            
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

        assert Matrix((A**0.0).trace()) == A.shape[0]
        """
        return NotImplemented

    @decorate.prepare(lambda self,other:(self,format(other)))
    @decorate.MultiMethod    
    def __rmul__(self,other):
        """
        other*self
        
        multiplication order doesn't matter for constants
        
            >>> assert K1*A == A*K1
        
            but it matters a lot for Matrix/Mvar multiplication
        
            >>> assert isinstance(A*M,Mvar)
            >>> assert isinstance(M.H*A,Matrix)
        
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
            >>> assert M.H*A==M.H*A.transform()

        Mvar*constant==constant*Mvar
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

    
    @__mul__.register(Mvar)
    @__rmul__.register(Mvar)
    def _scalarMul(self,scalar):
        """
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

            >>> assert 1j*A*1j==-A
        """
        return Mvar(
            mean= scalar*self.mean,
            var = scalar*self.var,
            vectors = self.vectors,
            square = not numpy.isreal(scalar),
        )

    @__mul__.register(Mvar,Matrix)
    def _matrixMul(self,matrix):
        """
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
        return Mvar(
            mean=self.mean*matrix,
            var=self.var,
            vectors=self.vectors*matrix,
        )

    @__rmul__.register(Mvar,Matrix)
    def _rmatrixMul(self,matrix):
        return matrix*self.transform()

    @__mul__.register(Mvar,Mvar)
    @__rmul__.register(Mvar,Mvar)
    def _mvarMul(self,mvar):
        """
        self*mvar

        multiplying two Mvars together is defined to fit with power
        
        >>> assert A*A==A**2
        >>> assert A*A==A*A.transform() or A.flat
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

    @__mul__.register(Mvar,numpy.ndarray)
    @__rmul__.register(Mvar,numpy.ndarray)
    def __vectorMul__(self,vector):
        assert (vector.ndim == 1), 'vector multiply, only accepts 1d arrays' 
        assert (vector.size == 1 or  vector.size == self.ndim),'vector multiply, vector.size must match mvar.ndim'
        
        return Mvar(
            mean=numpy.multiply(self.mean,vector),
            vectors=numpy.multiply(self.vectors,vector),
            var=self.var,
        )
            

    def quad(self,matrix=None):
        #todo: Chi & Chi2 distribution gives the *real* distribution of the length & length^2
        #       this gust has the right mean and variance
        """
        ref: http://en.wikipedia.org/wiki/Quadratic_form_(statistics)

        when used without a transform matrix this will get you the distribution 
        of the vector's magnitude**2.

        use this to dot an mvar with itself like (rand())**2 
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
        flattened   = transformed*self.mean.H

        return Mvar(
            mean=flattened.mean+numpy.trace(self.cov if matrix is None else matrix*self.cov) ,
            var=4.0*flattened.var+2.0*numpy.trace(transformed.cov*self.cov),
        )

    #todo: add a test case to show why quad and dot are different
    #todo: add a 'transposed' class so inner is just part of multiply

    @__mul__.register(Mvar,Mvar.T)
    def inner(self,other):
        """
        >>> assert A.inner(B) == B.inner(A)

        use this to dot product two mvars together, dot is like rand()*rand()
        be careful dot producting something with itself: 
            there you might want (rand())**2
            (use mvar.quad for that)
        """        
        return Mvar(
            mean=self.mean*other.mean.H,
            var=(
                (self*other).trace() + 
                (other*self.mean.H).trace() + 
                (self*other.mean.H).trace()
            )
        )
    
    dot = inner

    #todo: add a 'transposed' class so outer is just part of multiply
    @__mul__.register(Mvar.T,Mvar)
    def outer(self,other):
        """
        >>> assert(numpy.trace(A.outer(B))) == A.inner(B).mean
        """
        return numpy.outer(self.mean,other.mean)


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

        use arraymultiply as a shortcut for and diagonal matrix multiply
            >>> assert -1*A==-A
            >>> assert -A != A*(-E)
            >>> assert A*(-E) == [-1]*A

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
    def density(self,locations):
        """
        self(locations)

        Returns the probability density in the specified locations, 
        The vectors should be aligned onto the last dimension
        That last dimension is squeezed out during the calculation
        """
        return numpy.exp(-self.info(locations)) 
    
    def info(self,data=None,base=numpy.e):
        """
        >>> S=A.sample(100)   
        ... assert Matrix(-numpy.log(A(S))) == A.info(S)
        """
        if data is None:
            data = self

        if isinstance(data,Mvar):
            pass #!
        else:
            baseE=(
                self.dist2(data)+
                self.ndim*numpy.log(2*numpy.pi)+
                numpy.log(self.det())
            )/2

        return baseE/numpy.log(base)
        

    def entropy(self,base=numpy.e):
        """
        default is in base e

        >>> m=numpy.random.randn()
        ... assert A.entropy()+numpy.log(abs(numpy.linalg.det(m))) == (A*(m)).entropy()

        >>> assert (
        ...     numpy.array([
        ...         A[dim].entropy() 
        ...         for dim in range(A.ndim)
        ...     ]).sum() >= A.entropy() 
        ... )

        http://en.wikipedia.org/wiki/Multivariate_normal_distribution
        """
        baseE= numpy.log(self.det()*(2*numpy.pi*numpy.e)**self.ndim)/2
        return baseE/numpy.log(base)

    def KLdiv(self,other,base=numpy.e):
        """
        default is in base e

        >>> assert A.KLdiv(A)== 0
        >>> assert A.KLdiv(B) > 0

        http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        """
        baseE= (
            (self/other).trace()+
            ((other**-1)*(other.mean-self.mean).H).cov-
            numpy.log(self.det()/other.det())-
            self.ndim
        )/2
        return baseE/numpy.log(base)


    def __repr__(self):
        """
        print self
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
        
        #lazy import
        try:
            E=Ellipse
        except NameError:
            from matplotlib.patches import Ellipse as E
            globals().update({'Ellipse':E})


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
        return E(
            #with the Mvar's mean at the centre 
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
    
    return Mvar.fromCov(
        mean=(P.mean + yk*Kk.H),
        cov=(Matrix.eye(P.ndim)-Kk)*P.cov
    )

def givenVector(self,index,value):
    """
    direct implementation of the "given" algorithm in
    Andrew moore's data-mining/gussian slides

    >>> assert givenVector(A,index=0,value=1)==A.given(index=0,value=1)
    """
    fixed=binindex(index,self.ndim)
    if fixed.all():
        return Mvar.fromData(value)

    free=~fixed

    Mu = self[free]
    Mv = self[fixed]
    #todo: cleanup
    u=self.vectors[:,free]
    v=self.vectors[:,fixed]

    uv = self.var*numpy.array(u.H)*v

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
    
    return Mvar(
        mean=mean,
        vectors=vectors,
        var=result.var
    )



def mooreChain(self,sensor,transform=None):
        """
        given a distribution of actual values and an Mvar to act as a sensor 
        this method returns the joint distribution of real and measured values

        the, optional, transform parameter describes how to transform from actual
        space to sensor space
        """

        if transform is None:
            transform=Matrix.eye(self.ndim)

        T=(self*transform+sensor)
        vv=self.cov        

        return Mvar.fromCov(
            mean=numpy.hstack([self.mean,T.mean]),
            cov=numpy.vstack([
                numpy.hstack([vv,vv*transform]),
                numpy.hstack([(vv*transform).H,T.cov]),
            ])
        )



def binindex(index,numel):
    """
    convert an index to binary so it can be easily inverted
    """
    if hasattr(index,'dtype') and index.dtype==bool:
        return index
    
    binindex=numpy.zeros(numel,dtype=bool)
    binindex[index]=True

    return binindex



if __name__ == '__main__':
    #overwrite everything we just created with the copy that was 
    #created when we imported mvar; there can only be one.
    #from testObjects import *

    N1=1000
    N2=10
    data1 = numpy.random.randn(N1,2)+5*numpy.random.randn(1,2)
    data2 = numpy.random.randn(N2,2)+5*numpy.random.randn(1,2)
    
    A = Mvar.fromData(data1)
    B = Mvar.fromData(data2)

    print Mvar.fromData([A,B],Matrix.zeros) 
    print Mvar.fromData(A,Matrix.zeros)+Mvar.fromData(B,Matrix.zeros)

