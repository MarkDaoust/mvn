#! /usr/bin/env python

"""
This module contains only two things: the "Mvar" class, and the "wiki" 
function.

Mvar is the main idea of the package: Multivariate normal distributions 
    packaged to act like a vector. Perfect for kalman filtering, sensor fusion, 
    (and maybe expectation maximization)  

wiki is just to demonstrate the equivalency between my blending algorithm, 
    and the wikipedia version of it.
   (http://en.wikipedia.org/wiki/Kalman_filtering#Update)

The docstrings are full of examples. The test objects are created by runTest.sh, 
and stored in test_objects.pkl. You can get the most recent versions of them by 
importing testObjects.py, which will give you a module containing the objects used
in the tests
    A,B and C are instances of the Mvar class  
    K1 and K2 are random complex numbers
    M and M2 are complex valued matrixes
    E is an apropriately sized eye matrix
    N is an integer

remember: circular logic works because circluar logic works.
    a lot of the examples are demonstrations of what the code is doing, or expected
    invariants. they don't prove I'm right, but only that I'm being consistant
 

see their individual documentions for more information.    
"""

##imports


#self!
if __name__=='__main__':
    import mvar  

#builtins
import itertools
import collections 
import copy
import operator

#3rd party
import numpy

#maybe imports: third party things that we can live without
from maybe import Ellipse

#local
import helpers

from square import square
from automath import Automath
from right import Right
from inplace import Inplace
from matrix import Matrix

class Mvar(object,Automath,Right,Inplace):
    """
    Multivariate normal distributions packaged to act like a vector 
    (Ref: http://en.wikipedia.org/wiki/Vector_space)
    
    The class fully supports complex numbers.
    
    basic math operators (+,-,*,/,**,&) have been overloaded to work 'normally'
        for kalman filtering and common sense. But there are several surprising 
        features in the math these things produce, so watchout. 
        
        It basically boils down to the fact that there are differences between scalar, 
        matrix, and mvar operations, and if re-aranging an equation changes which type of 
        operation is called, the results will be different.
    
    This is perfect for kalman filtering, sensor fusion, or anything where you 
        need to track linked uncertianties across multiple variables 
        (
            like maybe the expectation maximization algorithm 
            and principal component analysis
            http://en.wikipedia.org/wiki/Expectation-maximization_algorithm
        )

    since the operations are defined for kalman filtering, the entire process 
        becomes:
        
        state[t+1] = (state[t]*STM + noise) & measurment
        
        Where 'state' is a series of mvars (indexed by time), 'noise' and 
        'measurment' are Mvars, ('noise' having a zero mean) and 'STM' is the 
        state transition matrix
        
    Sensor fusion is just:
        result = measurment1 & measurrment2 & measurment3
        or
        result = Mvar.blend(*measurments)
        
    normally (at least in what I read on wikipedia) these things are handled 
        with mean and covariance, but I find mean, variance, and eigenvectors 
        to be more useful, so that is how the data is actually managed in this 
        class, but other useful info in accessable through virtual attributes 
        (properties).
    
        This system makes compression (like principal component analysis) much 
        easier and more useful. Especially since, I can calculate the eigenvectors 
        without necessarily calculating the 
        covariance matrix
    
    actual attributes:
        mean
            mean of the distribution
        var
            the variance asociated with each vector.
        vectors
            unit vectors, as rows, not necessarily orthogonal. 
            only guranteed to give the right covariance see below.
        
        >>> assert A.vectors.H*numpy.diagflat(A.var)*A.vectors == A.cov
        
    virtual attributes (properties):
        ndim
            >>> assert A.ndim == A.mean.size

        cov
            gets or sets the covariance matrix
        scaled
            gets the vectors, scaled by one standard deviation
            (transforms from unit-eigen-space to data-space) 
        transform
            >>> assert A.transform()**2 == abs(A).cov 
            >>> assert A.transform()**N == A.transform(N)

            >>> assert A.transform(2) == abs(A).cov
            
            this is just more efficient than square-rooting the covariance, 
            since it is stored de-composed
            (transforms from unit-data-space to data-space) 
            
    
    the from* functions all create new instances from varous 
    common constructs.
        
    the get* functions all grab useful things out of the structure
    
    the inplace operators (like +=) work but, unlike in many classes, 
    do not currently speed up any operations.
    
    the mean of the distribution is stored as a row vector, so make sure align 
    your transforms apropriately and have the Mvar on left the when attempting 
    to do a matrix multiplies on it. This is for two reasons: 

        1) inplace operators work nicely (Mvar on the left)
        
        2) The Mvar is (currently) the only object that knows how to do 
        operations on itself, might as well go straight to it instead of 
        passing around 'NotImplemented's 
        
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
        
        var= var if callable(var) else numpy.array(var).flatten()[:,numpy.newaxis]
        mean= mean if callable(mean) else numpy.array(mean).flatten()[numpy.newaxis,:]
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

    def sign(self):
        return sign(self.var)

    def squeeze(self,**kwargs):
        """
        drop any vector/variance pairs with sqrt(variance) under the tolerence 
        limits the defaults match numpy's for 'allclose'
        """
        result=self.copy()
        (result.var,result.vectors)=helpers.squeeze(
            vectors=result.vectors,
            var=result.var
        )

        return result
            
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
    def from_data(data, bias=False, **kwargs):
        """
        >>> assert Mvar.from_data(A)==A 
        
        bias is passed to numpy's cov function.
        
        any kwargs are just passed on the Mvar constructor.
        
        this creates an Mvar with the same mean and covariance as the supplied 
        data with each row being a sample and each column being a dimenson
        
        remember numpy's default covariance calculation divides by (n-1) not 
        (n) set bias = 1 to use N,
        """
        if isinstance(data,Mvar):
            return data.copy()
        
        #get the number of samples, subtract 1 if un-biased
        N=data.shape[0] if bias else data.shape[0]-1
        
        #get the mean of the data
        mean=numpy.mean(data,axis=0)
        
        #calculate the covariance
        data-=mean
        data=Matrix(data)
        
        cov=(data.H*data)/N
        
        #create the mvar from the mean and covariance of the data
        return Mvar.fromCov(
            cov = cov,
            mean= mean,
            **kwargs
        )
        
    ############ get methods/properties

    def getCov(self):
        """
        get the covariance matrix used by the object
        
        >>> assert A.cov==A.vectors.H*numpy.diagflat(A.var)*A.vectors
        >>> assert A.cov==A.getCov()
        >>> assert A.scaled.H*A.scaled==abs(A).cov
        """
        return self.vectors.H*numpy.diagflat(self.var)*self.vectors
    
    cov = property(
        fget=getCov, 
        fset=lambda self,cov:self.copy(
            Mvar.fromCov(
                mean=self.mean,
                cov=cov,
        )),
        doc="set or get the covariance matrix"
    )
    
    scaled = property(
        fget=lambda self:Matrix(numpy.diagflat(self.var**(0.5+0j)))*self.vectors,
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
            >>> assert A*B.transform() == A*B  
        """
        if not numpy.isreal(self.var).all() or not(self.var>0).all():
            power = complex(power)

        if helpers.approx(power):
            self=self.inflate()
            vectors=self.vectors
            varP=numpy.ones_like(self.var)
            keep=slice(None)
        else:
            if numpy.real(power)<0:
                self=self.inflate()
                keep=~helpers.approx(self.var**(-1))
            else:                    
                keep=~helpers.approx(self.var)

            #don't involve the variances that will be infinite
            #(the power's already been divided by 2 above) 
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
            vectors, the secondis their lengths: the number of dimensions of 
            the space they are embedded in
            
            >>> assert A.vectors.shape == A.shape
            >>> assert (A.var.size,A.mean.size)==A.shape
            >>> assert A.shape[1]==A.ndim
        """
    )

    ########## Utilities
    def copy(self,other=None,deep=False):
        """
        either return a copy of an Mvar, or copy another into the self
        the default uses deep=False.
        >>> A2=A.copy(deep=True)        
        >>> assert A2 == A
        >>> assert A2 is not A
        >>> assert A2.mean is not A.mean
        >>> assert A2.var is not A.var
        >>> assert A2.vectors is not A.vectors

        >>> A.copy(B,deep=True)
        >>> assert B == A
        >>> assert B is not A
        >>> assert A.mean is not B.mean
        >>> assert A.var is not B.var
        >>> assert A.vectors is not B.vectors

        set deep=False to not copy the attributes
        >>> A2=A.copy(deep=False)        
        >>> assert A2 == A
        >>> assert A2 is not A
        >>> assert A2.mean is A.mean
        >>> assert A2.var is A.var
        >>> assert A2.vectors is A.vectors

        >>> A.copy(B,deep=False)
        >>> assert B == A
        >>> assert B is not A
        >>> assert A.mean is B.mean
        >>> assert A.var is B.var
        >>> assert A.vectors is B.vectors

        """ 
        C=copy.deepcopy if deep else copy.copy
        if other is None:
            return C(self)
        else:
            self.__dict__.update(C(other.__dict__))
        
    @staticmethod
    def stack(*mvars,**kwargs):
        """
        it's a static method to make it clear that it's not happening in place
        
        Stack two Mvars together, equivalent to hstacking the means, and 
        diag-stacking the covariances
        
        yes it works but be careful. Don't use this for reconnecting 
        something you calculated from an Mvar, back to the same Mvar it was 
        calculated from, you'll loose all the cross corelations. 
        If you're trying to do that use a better matrix multiply. 
        """
        #no 'square' is necessary here because the rotation matrixes are in 
        #entierly different dimensions
        return Mvar(
            #stack the means
            mean=numpy.concatenate(mvar.mean for mvar in mvars),
            #stack the vector diagonally
            vectors=diagstack(mvar.vectors for mvar in mvars),
            var=numpy.concatenate(mvar.var for var in mvars),
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
            helpers.ascomplex(randn(n,self.ndim,2))/sqrt(2)
            if cplx else 
            randn(n,self.ndim)
        )
        return Matrix(numpy.array(units*self.scaled.T)+self.mean)

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
        shape=A.shape
        return (
            0 if 
            shape[0]!=shape[1] else 
            numpy.prod(A.var)
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
        http://en.wikipedia.org/wiki/Quadratic_form_(statistics)
        """
        assert 1==0


    def dist2(self,locations):
        """
        return the square of the mahalabois distance from the Mvar to each vector.
        the vectors should be along the last dimension of the array.

        >>> assert helpers.approx(
        ...     (A**0).dist2(numpy.zeros((1,ndim))),
        ...     helpers.mag2((A**0).mean)
        ... )
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
        
    def binindex(self,index):
        """
        convert whatever format index, for this object, to binary 
        so it can be easily inverted
        """
        if hasattr(index,'dtype') and index.dtype==bool:
            return index
        
        binindex=numpy.zeros(self.ndim,dtype=bool)
        binindex[index]=True

        return binindex

    def given(self,index,value):
        """
        return an mvar representing the conditional probability distribution, 
        given the values, on the given indexes
        ref: andrew moore/data mining/gussians/page 22

        >>> assert A.given(0,0).mean[0,0]==0
        >>> assert A.given(0,0).vectors[:,0]==numpy.zeros
        """
        #convert the indexes
        index=self.binindex(index)
        
        #get the dimensions we'll need
        ndim=self.ndim

        #create the mean, 
        mean=numpy.zeros([1,ndim])
        #and set the values of interest
        mean[0,index]=value

        #invert the indexes
        nindex=~index
        #find the number of infinite vectors we need
        nvec=nindex.sum()

        vectors=numpy.zeros([nvec,ndim])
        where=numpy.argwhere(nindex)
        N=numpy.arange(nvec)[:,numpy.newaxis]

        vectors[N,where]=1

        var=numpy.inf*numpy.ones(nvec)

        return self & Mvar(
            mean=mean,
            vectors=vectors,
        ) 
        
        

    def __setitem__(self,index,value):
        """
        self[index]=value

        opertor interface to self.given 
        """
        self.copy(self.given(index,value))

    def marginal(self,index,squeeze=True, square=True):
        """
        return the marginal distribution in only the indexed dimensions
        return the marginal distribution,
        flattened in the indexed dimensions,
        unused mean components get replaced with nan's
        no dimensions are removed, be explicit, use knockout(index), or del()

        >>> flat = A[0].square().squeeze()
        >>> assert A[0].knockout(range(1,A.ndim)) == flat*flat.vectors.H
        """
        assert ~isinstance(index,tuple)

        index=self.binindex(index)

        result=self.copy(deep=True)
        result.mean[:,~index]=0
        result.vectors[:,~index]=0

        if (~index).all():
            result.var[:]=0
        elif square:
            result=result.square()

        if squeeze:
            result=result.squeeze()

        return result

    def __getitem__(self,index):
        """
        self[index]
        operator interface to self.marginal
        """
        return self.marginal(index)

    def knockout(self,index):
        """
        return an Mvar with the selected dimensions removed
        """
        keep=~self.binindex(index)
        return Mvar(
            mean=self.mean[:,keep],
            vectors=self.vectors[:,keep],
            var=self.var,
        )

    def __delitem__(self,index):
        """
        del(self[index])
        just an in-place interface to self.knockout
        """
        self.copy(self.knockout(index))

    ############ Math

    def __eq__(self,other):
        """
        self == other

        mostly it does what you would expect

        >>> assert A==A.copy()
        >>> assert A is not A.copy()
        >>> assert A != B

        but there are always surpriese.

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

        #check the number of dimensions of the space
        assert (
            self.ndim == other.ndim,
            """if the objects have different numbers of dimensions, you're doing something wrong"""
        )

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
    

    def blend(*mvars):
        """        
        This is awsome.
        
        optimally blend together any number of mvars, this is done with the 
        'and' operator because the elipses look like ven-diagrams
        
        Just choosing an apropriate inversion operator (1/A) allows us to 
        define kalman blending as a standard 'paralell' operation, like with 
        resistors. operator overloading takes care of the rest.
        
        The inversion automatically leads to power, multiply, and divide  
        
        When called as a method 'self' is part of *mvars 
        
        This blending function is not restricted to two inputs like the basic
        (wikipedia) version. Any number works,and the order doesn't matter.
        
        >>> assert A & B == B & A 
        >>> assert A & B == 1/(1/A+1/B)
        
        >>> abc=numpy.random.permutation([A,B,C])
        >>> assert A & B & C == helpers.paralell(*abc)
        >>> assert A & B & C == Mvar.blend(*abc)
        
        >>> assert (A & B) & C == A & (B & C)
        
        >>> assert (A & A).cov == A.cov/2
        >>> assert (A & A).mean == A.mean
        
        >>> assert A &-A == Mvar(mean=numpy.zeros(ndim))**-1
        >>> assert A &~A == Mvar(mean=numpy.zeros(ndim))**-1
        
        The proof that this is identical to the wikipedia definition of blend 
        is a little too involved to write here. Just try it (see the "wiki "
        function)
        
        >>> assert A & B == wiki(A,B)

        this algorithm is also, at the same time, solving linear equations
        where zero variances corespond to 

        >>> L1=Mvar(mean=[1,0],vectors=[0,1],var=numpy.inf)
        >>> L2=Mvar(mean=[0,1],vectors=[1,0],var=numpy.inf) 
        >>> assert (L1&L2).mean==[1,1]
        >>> assert (L1&L2).var.size==0
        """
        return reduce(operator.and_,mvars)
        
    def __and__(self,other):
        """
        self & other
        """
        #assuming the mvars are squared and squeezed 
        #if they both fill the space        
        if (
            self.shape[0] == self.shape[1] and 
            other.shape[0]==other.shape[1]
        ):
            #then this is a standard paralell operation
            return (self**(-1)+other**(-1))**(-1) 
        
        #otherwise there is more work to do
        
        #invert each object        
        Iself=self**-1
        Iother=other**-1

        Fself=numpy.isfinite(Iself.var)
        Fother=numpy.isfinite(Iother.var)

        #the object's null vectors will show up as having infinite 
        #variances in the inverted objects
        Nself=Iself.vectors[~Fself,:]
        Nother=Iother.vectors[~Fother,:] 

        #get length of the component of the means along each null vector
        Mself=Nself*self.mean.H
        Mother=Nother*other.mean.H

        #stack
        augNull = helpers.autostack([
            [Nself,Mself],
            [Nother,Mother],
        ])

        #square the stack
        (var,vec)=square(vectors=augNull)   

        #squeeze the stack
        (var,augNull) = helpers.squeeze(vectors=vec,var=var)
        
        #unstack it
        Nnew=augNull[:,:-1]
        Mnew=augNull[:,-1]

        Dmean=Mnew.H*numpy.diagflat(helpers.mag2(Nnew)**-1)*Nnew

        new=(Iself+Iother+Mvar(vectors=Nnew,var=Matrix.infs))**(-1)

        #add in the components of the means along each null vector
        new.mean=new.mean+Dmean

        return new
        

    def __pow__(self,power):
        """
        self**power

        >>> assert A**K1.real==A*A.transform(K1.real-1)

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
            >>> assert A == (A**-1)**-1
    
            >>> assert A**0 == A**(-1)*A
            >>> assert A**0 == A*A**(-1)
            >>> assert A**0 == A/A  
            >>> assert A**0*A==A
            >>> assert A*A**0==A

            >>> (A**K1)*(A**K2)==A**(K1+K2)
            False
            >>> A**K1/A**K2==A**(K1-K2)
            False

            those only work if the k's are real            
            >>> K1=K1.real
            >>> K2=K2.real
            >>> assert (A**K1)*(A**K2)==A**(K1+K2)
            >>> assert A**K1/A**K2==A**(K1-K2)
            
        Zero power has some interesting properties: 
            
            The resulting ellipse is always a unit sphere, 
            the mean is wherever it gets stretched to while we 
            transform the ellipse to a sphere
              
            >>> assert Matrix((A**0).var) == numpy.ones
            >>> assert (A**0).mean == A.mean*(A**-1).transform()

            if there are missing dimensions the transform is irreversable so this stops working 
            >>> if A.shape[0] == A.ndim:
            ...     assert (A**0).mean == A.mean*A.transform(-1)
            
        derivation of multiplication from this is messy.just remember that 
        all Mvars on the right, in a multiply, can just be converted to matrix:
            
            >>> assert A*B==A*B.transform()
            >>> assert M*B==M*B.transform()
            >>> assert A**2==A*A==A*A.transform()
        """
        self=self.inflate() if numpy.real(power)<=0 else self
        
        return Mvar(
            mean=self.mean*self.transform(power-1),
            vectors=self.vectors,
            var=self.var**power,
            square=False
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
                
                The pure mvar case fails for slightly different reasons, visible below:
                    >>> assert (B**0).transform() == Matrix.eye
                    >>> assert A*B**0 == A
                    >>> A*(B**0+B**0)==A*B**0+A*B**0
                    False

                    because
                    >>> assert A*(B**0+B**0) == A*(2*B**0)   #here the mean is stretched to sqrt(2) times 
                    >>> assert (2*B**0).transform() == 2**0.5*(B**0).transform()            
                    >>> assert A*B**0 + A*B**0 == 2*A*B**0 == 2*A #here it is outright multiplied by 2

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
            
            >>> assert sum(itertools.repeat(A,N-1),A) == A*(N)

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
        return Mvar(
            mean= scalar*self.mean,
            var = scalar*self.var,
            vectors = self.vectors,
            square = False,
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
        >>> assert A*A==A*A.transform()
        >>> assert A*B == A*B.transform()

       >>> assert A*(B**2) == A*(B.cov)

        Note that the result does not depend on the mean of the 
        second mvar(!) (really any mvar after the leftmost mvar or matrix)
        """
        return self*mvar.transform()

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
    def __iter__(self):
        raise ValueError("Mvars are not iterable... for now?")

    def __call__(self,locations):
         """
        self(locations)

         Returns the probability density in the specified locations, 
         The vectors should be aligned onto the last dimension
         That last dimension is squeezed out during the calculation
 
         If spacial dimensions have been flattened out of the mvar the result is always 1/0
         since the probablilities will have dimensions of hits/length**ndim 
         """
         return numpy.exp(self.dist2(self,locations))/2/numpy.pi/self.det(self)**0.5

        
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
        ]).replace('dtype=','dtype=numpy.').replace('numpy.numpy','numpy')
        
    
    __str__=__repr__

    ################ Art
    def getPatch(self,nstd=2,**kwargs):
        """
            get a matplotlib Ellipse patch representing the Mvar, 
            all **kwargs are passed on to the call to 
            matplotlib.patches.Ellipse

            not surprisingly Ellipse only works for 2d data.

            the number of standard deviations, 'nstd', is just a multiplier for 
            the eigen values. So the standard deviations are projected, if you 
            want volumetric standard deviations I think you need to multiply by sqrt(ndim)
        """
        if self.ndim != 2:
            raise ValueError(
                'this method can only produce patches for 2d data'
            )
        
        #unpack the width and height from the scale matrix 
        width,height = nstd*numpy.diagflat(self.var**(0.5+0j))
        
        #return an Ellipse patch
        return Ellipse(
            #with the Mvar's mean at the centre 
            xy=tuple(self.mean.flatten()),
            #matching width and height
            width=width, height=height,
            #and rotation angle pulled from the vectors matrix
            angle=numpy.rad2deg(
                numpy.angle(helpers.ascomplex(self.vectors)).flatten()[0]
            ),
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
        >>> assert A**-1 == A*A**-2
        >>> assert A & B == (A*A**-2+B*B**-2)**-1

        >>> D = A*(A.cov)**(-1) + B*(B.cov)**(-1)
        >>> assert A & B == D*(D.cov)**(-1)
        >>> assert A & B == wiki(A,B)
    """
    yk=M.mean-P.mean
    Sk=P.cov+M.cov
    Kk=P.cov*Sk.I
    
    return Mvar.fromCov(
        mean=(P.mean + yk*Kk.H),
        cov=(Matrix.eye(P.ndim)-Kk)*P.cov
    )

def newBlend(A,B):
    """
    cleaned up implementation of the wikipedia blending algorithm
    
        >>> assert newBlend(A,B) == wiki(A,B)
    """
    E=Matrix.eye(A.ndim)

    totalCovI=(A.cov+B.cov)**-1

    partA=A.cov*totalCovI
    partB=B.cov*totalCovI

    return Mvar.fromCov(
        mean=A.mean*(E-partA) + B.mean*(E-partB),
        cov=B.cov*A.cov*totalCovI
    )

def mooreGiven(self,index,value):
    """
    direct implementation of the "given" algorithm in
    Andrew moore's data-mining/gussian slides

    todo: figure out why this doesn't work
 
    >>> #assert mooreGiven(A,0,0)==A.given(0,0)
    """
    I=self.binindex(index)
    Iu=numpy.where(~I)[0]
    Iv=numpy.where( I)[0]
 
    U=self[Iu]
    V=self[Iv]

    vu=V.vectors.H*numpy.diagflat(self.var)*U.vectors

    return Mvar.fromCov(
        mean=U.mean+(value-V.mean)*(V**-1).cov*vu,
        cov=U.cov-vu.H*(V**-1).cov*vu,
    )


if __name__=='__main__':    
    #overwrite everything we just created with the copy that was 
    #created when we imported mvar, so there's only one copy.
    from mvar import *

