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

see their documention for more information.    
"""

##imports

#builtins
import itertools
from itertools import izip as zip
import collections 

import operator

#3rd party
import numpy

#maybe imports: third party things that we can live without
from maybe import Ellipse

#local
from helpers import autostack,diagstack,astype,paralell
from helpers import close,dots,rotation2d,isdiag,sortrows

from square import square

from automath import Automath
from inplace import Inplace
from matrix import Matrix


class Mvar(object,Automath,Inplace):
    """
    Multivariate normal distributions packaged to act like a vector 
    (http://en.wikipedia.org/wiki/Vector_space)
    
    The class fully supports complex numbers.
    
    basic math operators (+,-,*,/,**,&) have been overloaded to work 'normally'
        for kalman filtering and common sense. But there are several surprising 
        features in the math these things produce, so watchout.
    
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
        
        Where 'state' is a list of mvars (indexed by time), 'noise' and 
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
    
        This system make compression (like principal component analysis) much 
        easier and more useful. Especially since, in some of the newer code I 
        can calculate the eigenvectors withoug necessarily calculating the 
        covariance matrix
    
    actual attributes:
        mean
            mean of the distribution
        var
            the variance asociated with each vector.
        vectors
            unit vectors, as rows, not necessarily orthogonal. 
            only guranteed to give the right covariance see below.
        
        >>> assert A.vectors.H*A.var*A.vectors == A.cov
        
    virtual attributes (properties):
        cov
            gets or sets the covariance matrix
        scaled
            gets the vectors, scaled by one standard deviation
            (transforms from unit-eigen-space to data-space) 
        transform
            assert self.transform == (self.cov)**0.5 
            (transforms from unit-data-space to data-space) 
            
    
    the from* functions all create new instances from varous 
    common constructs.
        
    the get* functions all grab useful things out of the structure
    
    the do* functions all modify the structure inplace
    
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
        vectors=numpy.zeros,
        var=numpy.ones,
        mean=numpy.zeros,
        do_square=True,
        do_compress=True,
        **kwargs
    ):
        """
        Create an Mvar from available attributes.
        
        vectors: defaults to zeros
        var: (variance) defaults to ones
        
        >>> assert A.vectors.H*A.var*A.vectors == A.cov
        
        mean: defaults to zeros
        
        do_square:
            calls self.do_square() on the result if true. This sets the 
            vectors to orthogonal and unit length.
            
        do_compress
            calls self.do_compress() on the result if true. To clear out any 
            low valued vectors. It uses the same defaults as numpy.allclose()

        **kwargs is only used to pass in non standard defaults to the call to 
            do_compress, which is similar to numpy.allclose, 
            defaults are rtol=1e-5, atol=1e-8
        """
        #stack everything to check sizes and automatically inflate any 
        #functions that were passed in
        stack=Matrix(autostack([
            [var,vectors],
            [1  ,mean   ],
        ]))
        
        #unpack the stack into the object's parameters
        self.mean = numpy.array(stack[-1,1:]).flatten()
        self.var = numpy.array(stack[:-1,0]).flatten()
        self.vectors = stack[:-1,1:]
        
        assert numpy.all(isreal(var)),"variances must be real"
        
        if do_square:
            self.do_square()
        
        if do_compress:
            self.do_compress(**kwargs)
        
    def do_square(self):
        """
        this is NOT x**2 it is to set the vectors to perpendicular and unit 
        length
        """
        (self.var,self.vectors)=square(self.scaled);
        
    def do_compress(self,**kwargs):
        """
        drop any vector/variance pairs with variance under the tolerence limits
        the defaults match numpy's for 'allclose'
        """
        #convert the variance to a column vector
        var=self.var[:,numpy.newaxis]
        #align the variances with the vectors
        stack=numpy.hstack([var,self.vectors])
        #find wihich elements are close to zero
        C=numpy.array(close(stack[:,0],**kwargs)).squeeze()
        
        #keep only the var/vector pairs where the variance is not close to zero
        #and sort by variance
        stack=sortrows(stack[~C,:],column=0)

        #unstack them
        self.var = numpy.array(stack[:,0]).flatten()
        self.vectors = Matrix(stack[:,1:])
    
    ############## alternate creation methods
    @staticmethod
    def from_cov(cov,**kwargs):
        """
        everything in kwargs is passed directly to the constructor
        """
        assert cov==cov.H,'Covariance matrixes must be Hermitan'
        #get the variances and vectors.
        (var,vectors) = numpy.linalg.eigh(cov)
        vectors=Matrix(vectors.H)
        
        return Mvar(
            vectors=vectors,
            var=var,
            do_square=false,
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
        cov=(data.H*data)/N
        
        #create the mvar from the mean and covariance of the data
        return Mvar.from_cov(
            cov = cov,
            mean= mean,
            **kwargs
        )
        
     ############ get methods/properties

    def get_cov(self):
        """
        get the covariance matrix used by the object
        
        >>> assert A.cov==A.vectors.H*numpy.diagflat(A.var)*A.vectors
        >>> assert A.cov==A.get_cov()
        >>> assert A.scaled.H*A.scaled=abs(A).cov
        """
        return self.vectors.H*numpy.diagflat(self.var)*self.vectors
    
    cov = property(
        fget=get_cov, 
        fset=lambda self,cov:self.copy(
            Mvar.from_cov(
                mean=self.mean,
                cov=cov,
        )),
        doc="set or get the covariance matrix"
    )
    
    scaled = property(
        fget=lambda self:Matrix(numpy.diagflat(self.var**(0.5+0j)))*self.vectors,
        doc="""
            get the vectors, scaled by the standard deviations. 
            Useful for transforming from unit-eigen-space, to data-space
        """
    )
    
    transform = property(
        fget=lambda self:(
            self.vectors.H*
            Matrix(numpy.diagflat(self.var**(0.5+0j)))*
            self.vectors
        ),
        doc="""
            get the get the un-rotate*scaled transform, 
            Useful for transforming from unit-data-space, to data-space
        """
    )


    ########## Utilities
    def copy(self,other=None):
        """
        either return a copy of an Mvar, or copy another into the self
        B=A.copy()
        A.copy(B)
        """ 
        if other is None:
            return Mvar.from_attr(
                mean=self.mean,
                vectors=self.vectors,
                var=self.var
            )
        else:
            self.__dict__=other.__dict__.copy()
        
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
    
    def sample(self,n=1):
        """
        take samples from the distribution
        n is the number of samples, the default is 1
        each sample is a numpy matrix row vector.
        
        a large number of samples will have the same mean and cov as the 
        Mvar being sampled
        """
        assert all(self.var>0),(
            "you can't sample a distribution with negative variance"
        )
        
        data= Matrix(numpy.random.randn(n,self.mean.size))*self.scaled.T
        
        return Matrix(numpy.array(data)+self.mean)
    
    ############ Math

    #### logical operations
    def __eq__(self,other):
        """
        >>> assert A==A
        
        compares the means and covariances of the distributions
        """
        return Matrix(self.mean)==Matrix(other.mean) and self.cov==other.cov
    
    def __abs__(self):
        """
        sets all the variances to positive
        >>> assert numpy.all(A.var>=0)
        >>> assert abs(A) == abs(~A)
        >>> assert numpy.all(A.mean == abs(A).mean)
        """
        result=self.copy()
        result.var=numpy.abs(self.var)
        return result

    def __pos__(self):
        """
        it apears to do nothing
        >>> assert A == +A
        
        but it is also a shortcut to do_square
        >>> assert (+A).vectors*(+A).vectors.H==matrix.eye
        
        it does the squaring in place, then returns the copy.
        """
        self.do_square()
        return self.copy()
    
    def __invert__(self):
        """
        invert negates the covariance without negating the mean.
        >>> assert (~A).mean == A.mean
        >>> assert (~A).var == (-A).var == -(A.var)
        """
        result=self.copy()
        result.var=-(self.var)
        return result
    
    def blend(*mvars):
        """
        A & ?
        
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
        
        >>> abc=[A,B,C]
        >>> #the order doesn't matter, shuffle abc in place.
        >>> numpy.random.shuffle(abc)
        >>> assert A & B & C == paralell(*abc)
        >>> assert A & B & C == Mvar.blend(*abc)== Mvar.__and__(*abc)
        >>> assert A & B & C == A & (B & C)
        
        The proof that this is identical to the wikipedia definition of blend 
        is a little too involved to write here. Just try it (see the "wiki 
"        function)
        
        >>> assert A & B == wiki(A,B)
        """
        return paralell(*mvars)
        
    __and__ = blend

    ### operators
    def __pow__(self,power):
        """
        A**?
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        This definition was developed to turn kalman blending into a standard 
        resistor-style 'paralell' operation
        
        The main idea is that only the scale matrix (eigenvalues) gets powered.
        (which is normal for diagonalizable matrixes), stretching the sheet at 
        an independant rate along each (perpendicular) eigenvector
        
        Because the scale matrix is a diagonal, powers on it are easy, 
        so this is not restricted to integer powers
        
        But the mean is also affected by the stretching. It's as if the usual 
        value of the mean is a "zero power mean" transformed by whatever is 
        the current value of the A.vectors.H*A.scaled matrix and if you change that 
        the mean changes with it..
        
        Most things you expect to work just work.
        
            >>> assert A**0== A**(-1)*A== A*A**(-1)== A/A        
            >>> assert (A**K1)*(A**K2)==A**(K1+K2)
            >>> assert A**K1/A**K2==A**(K1-K2)
        
        Zero power h333333as some interesting properties: 
            
            The resulting ellipse is always a unit sphere, with the orientation 
            unchanged, but the mean is wherever it gets stretched to while we 
            transform the ellipse to a sphere
              
            >>> assert (A**0).scale==numpy.eye(A.mean.size)
            >>> assert (A**0).vectors== A.vectors
            >>> assert (A**0).mean == A.mean*(A**-1).transform == A.mean*A.transform**(-1)
            
        derivation of multiplication:
            >>> assert A.scaled == A.scale*A.vectors
            >>> assert (A**K1).scaled == numpy.diag(A.scale.diagonal()**K1)*A.vectors
            >>> scaled = lambda mvar:Matrix(numpy.diagflat(mvar.var**0.5))*mvar.scaled 
            >>> assert (
                (A**K1).scaled == 
                scaled(A)*A.vectors.H*Matrix(numpy.diagflat(A.var**((K1-1)/2))*A.scaled
            >>> assert (A**K1).mean == A.mean*A.vectors.H*numpy.diag(A.scale.diagonal()**(K1-1))*A.vectors
            
            that's a matrix multiply.
            
            So all Mvars on the right,in a multiply, can just be converted to 
            matrix:
            
            >>> assert A*B==A*B.vectors.H*B.scale*B.vectors
        """
        vectors = self.vectors
        new_scale = Matrix(
            numpy.diag(numpy.diagonal(self.scale)**(power-1))
        )
            
        transform = vectors.H*new_scale*vectors
        
        return self*transform
        
    def __mul__(
        self,
        other,
        #this function is applied to the right operand, to simplify the types 
        #of the objects we have to deal with, and to convert any Mvars on 
        #the left into a vectors.H*scale*vectors matrix.
        rconvert=lambda 
            item,
            helper=lambda item:(
                Matrix(item) 
                if item.ndim else
                item
            )
        :(  
            Matrix(item.vectors.H*item.scale*item.vectors) if 

            isinstance(item,Mvar) else 
            helper(numpy.array(item))
        ),
        #this dict is used to dispatch multiplications based on the type of 
        #the right operand, after it has been passed through rconvert
        multipliers={
            (Matrix):(
                lambda self,matrix:Mvar.from_attr(
                    mean=self.mean*matrix,
                    scaled=self.scale*self.vectors*matrix,
                )
            ),
            (numpy.ndarray):(
                lambda self,constant:Mvar.from_cov(
                    mean= constant*self.mean,
                    cov = constant*self.cov
                )
            )
        }    
    ):
        """
        A*?
        
        coercion notes:
            All non Mvar imputs will be converted to numpy arrays, then 
            treated as constants if zero dimensional, or matrixes otherwise 
            
            Mvar always beats constant. Between Mvar and Matrix the left 
            operand wins 
            
            >>> assert isinstance(A*B,Mvar)
            >>> assert isinstance(A*M,Mvar)
            >>> assert isinstance(M*A,Matrix) 
            >>> assert isinstance(A*K1,Mvar)
            >>> assert isinstance(A*N,Mvar)
            >>> assert isinstance(K1*A,Mvar)
            
            whenever an mvar is found on the right it is converted to a 
            vectors.H*scale*vectors matrix and the multiplication is 
            re-called.
            
        general properties:
            
            constants still commute            
            >>> assert K1*A*M == A*K1*M == A*M*K1
            
            but the asociative property is lost if you mix constants and 
            matrixes (but I think it's ok if you only have 1 of the two types?)
            
            >>> assert (A*4).cov == (A*(2*numpy.eye(2))).cov
            
            ????
            asociative if only mvars and matrixes?
            ????
            still distributive?
            
        Mvar*Mvar
            multiplying two Mvars together is defined to fit with power
            
            >>> assert A*A==A**2
            >>> assert (A*B).affine==(A*B.vectors.H*B.scaled).affine
            >>> assert (A*B).mean==A.mean*B.vectors.H*B.scale*B.vectors
            >>> assert A*(B**2) == A*(B.cov)
            
            Note that the result does not depend on the mean of the 
            second mvar(!) (really any mvar after the leftmost mvar or matrix)

        Mvar*constant == constant*Mvar
            Matrix multiplication and scalar multiplication behave differently 
            from eachother.  
            
            For this to be a properly defined vector space scalar 
            multiplication must fit with addition, and addition here is 
            defined so it can be used in the kalman noise addition step so: 
            
            >>> assert (A+A).scaled==(2*A).scaled
            >>> assert (A+A).scaled==(2**0.5)*A.scaled
            
            >>> assert (A+A).mean==(2*A).mean
            >>> assert (A+A).mean==2*A.mean
            
            >>> assert (A*K1).scaled==(K1**0.5)*A.scaled
            >>> assert (A*K1).mean==K1*A.mean
            
            >>> assert sum(itertools.repeat(A,N-1),A) == A*(N)
            
            >>> assert (A*K1).cov==A.cov*K1
            
            be careful with negative constants because you will end up with 
            imaginary numbers in your scaled matrix, as a direct result of:            
            
            assert (A*K1).scaled==(K1**0.5)*A.scaled
            assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
            
            if you want to scale the distribution linearily with the mean
            then use matrix multiplication
        
        Mvar*matrix
        
            matrix multiplication transforms the mean and ellipse of the 
            distribution. Defined this way to work with the kalman state 
            update step.
            
            simple scale is like this
            >>> assert (A*(numpy.eye(A.mean.size)*K1)).scaled==A.scaled*K1
            >>> assert (A*(numpy.eye(A.mean.size)*K1)).mean==A.mean*K1
            
            or more generally
            >>> assert (A*M).cov==M.H*A.cov*M
            >>> assert (A*M).mean==A.mean*M
            
            matrix multiplication is implemented as follows
            
            assert A*M == Mvar.from_affine(A.affine*diagstack([M,1]))
            
        given __mul__ and __pow__ it would be immoral to not overload divide 
        as well, automath takes care of these details
            A/?
            
            >>> assert A/B == A*(B**(-1))
            >>> assert A/M == A*(M**(-1))
            >>> assert A/K1 == A*(K1**(-1))
        
            ?/A: see __rmul__ and __pow__
            
            >>> assert K1/A == K1*(A**(-1))
            >>> assert M/A==M*(A**(-1))
        """
        other=rconvert(other)
        return multipliers[type(other)](self,other) 
    
    def __rmul__(
        self,
        other,
        #here we convert the left operand to a numpy.ndarray if it is a scalar,
        #otherwise we convert it to a Matrix.
        #the self (right operand) will stay an Mvar for scalar multiplication
        #or be converted to a vectors.H*scale*vectors matrix for matrix 
        #multiplication
        convert=lambda
            other,
            self,
            helper=lambda other,self: (
                Matrix(other) if 
                other.ndim else
                other
                ,
                Matrix(self.vectors.H*self.scale*self.vectors) if 
                other.ndim else
                self
            )
        :helper(numpy.array(other),self)
        ,
        #dispatch the multiplication based on the type of the left operand
        multipliers={
            #if the left operand is a matrix, the mvar has been converted to
            #to a matrix -> use matrix multiply
            (Matrix):operator.mul,
            #if the left operand is a constant use scalar multiply
            (numpy.ndarray):(
                lambda constant,self:Mvar.from_cov(
                    mean= constant*self.mean,
                    cov = constant*self.cov
                )
            )
        }
    ):
        """
        ?*A
        
        multiplication order:
            doesn't matter for constants
        
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
        
        Mvar*Mvar
            multiplying two Mvars together fits with the definition of power
            
            >>> assert B*B == B**2
            >>> assert A*B == A*B.vectors.H*B.scaled 
            
            the second Mvar is automatically converted to a matrix, and the 
            result is handled by matrix multiply
            
            again note that the result does not depend on the mean of the 
            second mvar(!)
        
        martix*Mvar
            >>> assert M*A==M*A.vectors.H*A.scale*A.vectors

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
        (other,self)= convert(other,self)
        return multipliers[type(other)](other,self)
    
    def __add__(self,other):
        """
        A+?
        
        When using addition keep in mind that rand()+rand() is not like scaling 
        one random number by 2, it adds together two random numbers.

        The add here is like rand()+rand()
        
        Addition is defined this way so it can be used directly in the kalman 
        noise addition step
        
        so if you want simple scale use matrix multiplication like rand()*(2*eye)
        
        scalar multiplication however fits with addition:
        
        >>> assert (A+A).scaled==(2*A).scaled
        >>> assert (A+A).scaled==(2**0.5)*A.scaled
        
        >>> assert (A+A).mean==(2*A).mean
        >>> assert (A+A).mean==2*A.mean
        
        >>> assert (A+B).mean==A.mean+B.mean
        >>> assert (A+B).cov==A.cov+B.cov
        
        watch out subtraction is the inverse of addition 
            >>> assert A-A == Mvar.from_attr(mean=[0,0])
            >>> assert (A-B)+B == A
            >>> assert (A-B).mean== A.mean - B.mean
            >>> assert (A-B).cov== A.cov - B.cov
            
        if you want something that acts like rand()-rand() use:
            
            >>> assert (A+B*(-1*numpy.eye(A.mean.size))).mean== A.mean - B.mean
            >>> assert (A+B*(-1*numpy.eye(A.mean.size))).cov== A.cov + B.cov

        __sub__ should also fit with __neg__, __add__, and scalar multiplication.
        
            >>> assert B+(-A) == B+(-1)*A == B-A
            >>> assert A-B == -(B-A)
            
            but watchout you'll end up with complex... everything?
        """
        #stack all the scaled
        stack=numpy.vstack((self.scaled,other.scaled))

        return Mvar.from_cov(
            mean= (self.mean+other.mean),
            #this next line is what fails my tests
            #solution? 
            #http://en.wikipedia.org/wiki/Square_root_of_a_matrix:
            #   (math notation converted to local python standard)
            #   """if T = A*A.H = B*B.H, then there exists a unitary U s.t. 
            #    A = B*U"""
            #
            #a unitary matrix is a complex rotation matrix
            #http://en.wikipedia.org/wiki/Unitary_matrix
            #   """In mathematics, a unitary matrix is an nxn complex matrix U 
            #    satisfying the condition U.H*U = I, U*U.H = I"""
            #
            #I don't know how to fix it but this is close. 
            cov = (stack.H*stack),
        )
        
    ################# Non-Math python internals
    def __iter__(self):
        scaled=self.scaled
        return iter(numpy.vstack(scaled,-scaled))
        
    def __repr__(self):
        return '\n'.join([
            'Mvar.from_attr(',
            '    mean=',8*' '+self.mean.__repr__().replace('\n','\n'+8*' ')+',',
            '    scale=',8*' '+self.scale.__repr__().replace('\n','\n'+8*' ')+',',
            '    vectors=',8*' '+self.vectors.__repr__().replace('\n','\n'+8*' ')+',',
            ')',
        ])
    
    __str__=__repr__

    ################ Art
    def get_patch(self,nstd=3,**kwargs):
        """
            get a matplotlib Ellipse patch representing the Mvar, 
            all **kwargs are passed on to the call to 
            matplotlib.patches.Ellipse

            not surprisingly Ellipse only works for 2d data.

            the number of standard deviations, 'nstd', is just a multiplier for 
            the eigen values. So the standard deviations are projected, if you 
            want volumetric standard deviations I think you need to multiply by sqrt(ndim)
        """
        if self.mean.size != 2:
            raise ValueError(
                'this method can only produce patches for 2d data'
            )
        
        #unpack the width and height from the scale matrix 
        width,height = nstd*numpy.diag(self.scale)
        
        #return an Ellipse patch
        return Ellipse(
            #with the Mvar's mean at the centre 
            xy=tuple(self.mean.flatten()),
            #matching width and height
            width=width, height=height,
            #and rotation angle pulled from the vectors matrix
            angle=numpy.rad2deg(
                numpy.angle(astype(
                    self.vectors,
                    complex,
                )).flatten()[0]
            ),
            #while transmitting any kwargs.
            **kwargs
        )


## extras    

def wiki(P,M):
    """
    Direct implementation of the wikipedia blending algorythm
    
    The quickest way to prove it's equivalent is by examining this:
        >>> assert A & B == ((A*A**-2)+(B*B**-2))**-1
    """
    yk=M.mean.H-P.mean.H
    Sk=P.cov+M.cov
    Kk=dots(P.cov,Matrix(Sk).I)
    
    return Mvar.from_cov(
        mean=(P.mean.H+dots(Kk,yk)).H,
        cov=dots((numpy.eye(P.mean.size)-Kk),P.cov)
    )

def isplit(sequence,fkey=bool): 
    """
        return a defaultdict (where the default is an empty list), 
        where every value is a sub iterator produced from the sequence
        where items are sent to iterators based on the value of fkey(item).
        
        >>> isodd = isplit(xrange(1,7),lambda item:bool(item%2))
        >>> isodd[True]
        [1, 3, 5]
        >>> isodd[False]
        [2, 4, 6]
        
        which gives the same results as
        
        >>> X=xrange(1,7)
        >>> [item for item in X if bool(item%2)]
        [1, 3, 5]
        >>> [item for item in X if not bool(item%2)]
        [2, 4, 6]
        
        or you could make a mess of maps and filter
        but this is so smooth,and really shortens things 
        when dealing with a lot of keys 
        
        >>> bytype = isplit([1,'a',True,"abc",5,7,False],type)
        >>> bytype[int]
        [1, 5, 7]
        >>> bytype[str]
        ['a', 'abc']
        >>> bytype[bool]
        [True, False]
        >>> bytype[dict]
        []
    """
    result = collections.defaultdict(list)
    for key,iterator in itertools.groupby(sequence,fkey):
        R = result[key]
        R.extend(iterator)
        result[key] = R
        
    return result

if __name__=="__main__":
    import doctest
    #create random test objects
    A=Mvar.from_attr(mean=10*astype(numpy.random.randn(1,2,2),complex),scaled=10*astype(numpy.random.randn(2,2,2),complex))
    B=Mvar.from_cov(mean=10*astype(numpy.random.randn(1,2,2),complex),cov=(lambda x:x*x.H)(Matrix(10*astype(numpy.random.randn(2,2,2),complex))))
    C=Mvar.from_data(dots(astype(numpy.random.randn(50,2,2),complex),10*astype(numpy.random.randn(2,2,2),complex)))
   
    
    M=Matrix(numpy.random.randn(2,2))
    
    K1=numpy.random.rand()+0j
    K2=numpy.random.rand()+0j
        
    N=numpy.random.randint(2,10)
    
    print 'from numpy import array'
    print 'from mvar import Mvar,Matrix'
    print '#test objects used'
    print 'A=',A
    print 'B=',B
    print 'C=',C
    print 'M=',M.__repr__()
    print 'K1=',K1
    print 'K2=',K2
    print 'N=',N
    
    doctest.testmod()

    

    
    
