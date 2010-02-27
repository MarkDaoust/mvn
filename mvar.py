#! /usr/bin/env python

##imports
#internals

#builtins
import itertools
from itertools import izip as zip
import collections 

import operator

#3rd party
import numpy

try:
    from matplotlib.patches import Ellipse
except ImportError:
    def Ellipse(*args,**kwargs):
        """
        Unable to find matplotlib.patches.Ellipse
        """
        raise ImportError(
            "Ellipse is required, from matplotlib.patches, to get a patch"
        )

#local
from helpers import autostack,diagstack,astype,paralell,close,dot,rotation2d
from automath import Automath
from inplace import Inplace

class Matrix(numpy.matrix):
    """
    Imporved version of the martix class.
    the only modifications are:
        division doesn't try to do elementwise division, it tries to multiply 
            by the inverse of the other
        __eq__ runs numpy.allclose, so the matrix is treated as one thing, not 
            a collection of things.
    """
    def __new__(cls,data,dtype=None,copy=True):
        self=numpy.matrix(data,dtype,copy)
        self.__class__=cls
        return self

    def __eq__(self,other):
        return numpy.allclose(self,other)
    
    def __div__(self,other):
        return self*other**(-1)

    def __rdiv__(self,other):
        return other*self**(-1)

    def __repr__(self):
        S=numpy.matrix.__repr__(self)
        return 'M'+S[1:]

    def diagonal(self):
        return numpy.squeeze(numpy.array(numpy.matrix.diagonal(self)))
    
    __str__ = __repr__


class Mvar(object,Automath,Inplace):
    """
    Multivariate normal distributions packaged to act like a vector.
        (it's an extension of the vectors)
    
    This is done with kalman filtering in mind, but is good for anything where 
        you need to track linked uncertianties across multiple variables.

    basic math operators (+,-,*,/,**,&) have been overloaded to work 'normally'
    for kalman filtering and common sense. But there are *a lot* of surprising 
    features in the math these things produce, so watchout.
    
    since the operations are defined for kalman filtering, the entire process 
    becomes:
        
        state[t+1] = (state[t]*STM + noise) & measurment
        
        state is a list of mvars (indexed by time), noise and measurment are 
        Mvars, (noise having a zero mean) and 'STM' is the state transition 
        matrix
        
    
    A nice side effect of this is that sensor fusion is just:
    
        result = measurment1 & measurrment2 & measurment3
       
        or
        
        result = paralell(*measurments)
        
    normally (at least in what I read on wikipedia) these things are handled 
    with mean and covariance, but I find mean,scale,rotation to be more useful, 
    so that is how the data is actually managed, but other useful info in 
    accessable through virtual attributes (properties).
    
    This system make compression (think principal component analysis) much 
    easier and more useful, but until I can think of a way to get directly 
    from data to the eigenvectors of the covariance matrix of the data, without 
    calculating the covariance, it is of limited utility).
    
    actual attributes:
        mean
            mean of the distribution
        scale
            the eigenvalue asociated with each eigenvector,as a diagonal matrix
        rotation
            unit eigenvectors, as rows
    virtual attributes:    
        vectors
            vectors eigenvectors, as rows
        cov
            covariance matrix
        affine
            autostack([
                [self.vectors,numpy.zeros],
                [self.mean   ,          1],
            ])
            
        
    the from* functions all create new instances from varous 
    common constructs.
        
    the get* functions all grab useful things out of the structure, 
    all have equivalent properties linked to them
    
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
    and speed actually is a problem it's not worth working on. 
    """
    
    ############## Creation
    def __init__(self,
        stack,
        do_square=True,
        do_compress=True,
        **kwargs
    ):
        """
        create the Mvar from the stack of attributes
        
        stack= autostack([
                [numpy.ones,vectors],
                [         1,   mean],
            ])
        
        stack= autostack([
                [scale,rotation],
                [    1,    mean],
            ])
        
        do_square 
            calls self.do_square() on the result if true. To set the rotation 
            to orthogonal and unit length, do_square automatically calls 
            do_compress, 
            
        do_compress
            calls self.do_compress() on the result if true. To clear out any 
            low valued vectors uses the same defaults as numpy.allclose()
        """
        stack=numpy.real_if_close(stack)
        
        #unpack the stack into the object's parameters
        self.mean = Matrix(stack[-1,1:])
        self.scale = Matrix(numpy.diagflat(stack[:-1,0]))
        self.rotation = Matrix(stack[:-1,1:])
        
        assert not do_square or do_compress,"do_square calls do_compress"
        
        if do_square:
            self.do_square()
        elif do_compress:
            self.do_compress()
        
    def do_square(self,**kwargs):
        """
        this is NOT x**2 it is to set the vectors to perpendicular and unit 
        length
        
        **kwargs is just passed on to do_compress
        """
        #to decouple compress from square you'll need do the square on the 
        #Mvar's brane instead of the full-space, to do that you'll need
        #something like the plane class I've started developing in the adjacent file
        V=self.rotation
        S=self.scale
    
        if not Matrix(dot(V.H,V))==Matrix(numpy.eye(V.shape[0])):
            (scale,rotation) = numpy.linalg.eig(dot(V.H,S,S,V))
            self.rotation=Matrix(rotation).H
            self.scale=Matrix(numpy.diagflat(scale**(0.5+0j)))
        
        self.do_compress(**kwargs)
        
    def do_compress(self,rtol=1e-5,atol=1e-8):
        """
        drop any vector/scale pairs which are under the tolerence limits
        the defaults match numpy's for 'allclose'
        """
        #convert the scale to a column vector
        diag=numpy.diagonal(self.scale)[:,numpy.newaxis]
        #get the rotation
        rotation=self.rotation
        stack=numpy.hstack([diag,rotation])
        stack=stack[numpy.argsort(diag.flatten()),:]
        C=~numpy.array(close(stack[:,0],rtol=rtol,atol=atol)).squeeze()
        #drop the scale/rotation where the scale is close to zero
        stack=stack[C,:]
        #unstack them
        self.scale = Matrix(numpy.diagflat(stack[:,0]))
        self.rotation = Matrix(stack[:,1:])
    
    ############## alternate creation methods
    @staticmethod
    def from_attr(
        mean = numpy.zeros,
        vectors = numpy.zeros, 
        scale = numpy.ones,
        **kwargs
    ):
        """
        create the Mvar from available arrtibutes, rotation isn't listed 
        becaues 'vectors' does everything rotation would 
        
        mean
            defaults to numpy.zeros
            
        vectors
            defaults to numpy.zeros. again row vectors. They do not need to be 
            orthogonal, or unit length.
            
        scale
            defaults to numpy.ones
            
        """
        if 'rotation' in kwargs:
            if vectors is numpy.zeros:
                vectors=kwargs['rotation']
                kwargs['do_square']=True
            else :
                raise TypeError('Supply rotation, vectors, or neither one. Not both')
        
        if not callable(scale):
            scale=numpy.array(scale)
            if isdiag(scale):
                scale=scale.diagonal()
            scale=scale.squeeze()
            assert scale.ndim==1,"scales must be flat or diagonal"
            scale=scale[:,numpy.newaxis]
        
        if not callable(mean):
            mean=numpy.array(mean).squeeze()[numpy.newaxis,:]
            
        #use autostack to determine unknown sizes
        return Mvar(
            autostack([
                [scale,vectors],
                [  1.0,   mean],
            ]),**kwargs
        )
    
    @staticmethod
    def from_cov(cov,**kwargs):
        """
        everything in kwargs is passed directly to the constructor
        don't bother to set 'do_square' to true, they will automatically 
        be orthogonal when pulled out of the covariance
        """
        #get the scale and rotation matracies
        scale,rotation = numpy.linalg.eig(cov)
        
        return Mvar.from_attr(
            vectors=Matrix(rotation).H,
            #square root the scales
            scale=numpy.real_if_close((scale)**(0.5+0j)),
            do_square=False,
            **kwargs
        )
    
    @staticmethod
    def from_data(data, bias=0, **kwargs):
        """
        >>> assert Mvar.from_data(A)==A 
        
        bias is passed to numpy's cov function.
        
        any kwargs are just passed on the Mvar constructor.
        
        this creates an Mvar with the same mean and covariance as the supplied data
        with each row being a sample and each column being a dimenson
        
        remember numpy's default covariance calculation divides by (n-1) not 
        (n) set bias = 1 to use N,
        
        """
        if isinstance(data,Mvar):
            return data.copy()
        
        #convert the data to a matrix 
        data=Matrix(data)
        #create the mvar from the mean and covariance of the data
        return Mvar.from_cov(
            cov = numpy.cov(data,bias=bias,rowvar=0),
            mean= numpy.mean(data,axis=0),
            **kwargs
        )
        
    def from_affine(affine,**kwargs):
        """
        unpack an affine transform, into an Mvar.
        the transform should be in the format below:
        
        autostack([
            [self.vectors,numpy.zeros]
            [self.mean   ,        1.0]
        ])
        """
        return Mvar(
            autostack([[numpy.ones,affine[:,:-1]]])
            **kwargs
        )

     ############ get methods/properties

    def get_cov(self):
        """
        get the covariance matrix used by the object
        
        >>> assert A.cov==dot(A.vectors.H,A.vectors)
        >>> assert A.cov==dot(A.rotation.H,A.scale,A.scale,A.rotation)
        """
        vectors=self.vectors
        return dot(vectors.H,vectors)
    
    def get_vectors(self):
        """
        get the matrix of scaled eigenvectors (as rows)

        >>> assert A.vectors==dot(A.scale,A.rotation)
        """
        return dot(self.scale,self.rotation)
    
    def get_affine(self):
        return  autostack([
            [self.vectors,numpy.zeros],
            [self.mean   ,          1],
        ])

    cov = property(
        fget=get_cov, 
        fset=lambda self,cov:self.copy(
            Mvar.from_cov(
                mean=self.mean,
                cov=cov,
    )))
    
    vectors = property(
        fget=get_vectors, 
        fset=lambda self,vectors:self.copy(
            Mvar.from_attr(
                mean=self.mean,
                vectors=vectors,
    )))
    
    affine = property(
        fget=get_affine,
        fset=lambda self,affine:self.copy(
            Mvar.from_affine(
                affine,
    )))
    
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
                rotation=self.rotation,
                scale=self.scale
            )
        else:
            self.__dict__=other.__dict__.copy()
        
    @staticmethod
    def stack(*mvars,**kwargs):
        """
        it's a static method to make it clear that it's not happening in place
        Stack two Mvars together, equivalent to hstacking the rotation, and 
        diagstacking the covariance matrixes
        
        yes it works but be careful. Don't use this for reconnecting 
        something you calculated from an Mvar, back to the same Mvar it was 
        calculated from, you'll loose all the cross corelations. 
        If you're trying to do that use a better matrix multiply. 
        """
        #no 'refresh' is necessary here because the rotation matrixes are in 
        #entierly different dimensions
        return Mvar.from_attr(
            #stack the means horizontally
            mean=numpy.hstack([mvar.mean for mvar in mvars]),
            #stack the vector packets diagonally
            rotation=diagstack([mvar.rotation for mvar in mvars]),
            scale=numpy.hstack([Matrix(mvar.scale) for scale in mvars]),
            **kwargs
        )
    
    def sample(self,n=1):
        """
        take samples from the distribution
        n is the number of samples, the default is 1
        each sample is a numpy matrix row vector.
        
        the samles will have the same mean and cov as the distribution 
        being sampled
        """
        data=numpy.hstack([
            numpy.random.randn(n,self.mean.size),
            numpy.ones([n,1]),
        ])
        
        transform=numpy.vstack([
            self.vectors,
            self.mean,
        ])

        return dot(data,transform)
    
    ############ Math

    #### logical operations
    def __eq__(self,other):
        """
        A==?
        compares the means and covariances or the distributions
        """
        return self.mean==other.mean and self.cov==other.cov
        
    def blend(*mvars):
        """
        A & ?
        
        This is awsome.
        
        optimally blend together any number of mvars, this is done in and 
        because the elipses look like ven-diagrams
        
        And just choosing an apropriate inversion operator (1/A) allows us to 
        define kalman blending as a standard 'paralell' operation, like with 
        resistors. operator overloading takes care of the rest.
        
        The inversion automatically leads to power, multiply, and divide  
        
        When called as a method 'self' is part of *mvars 
        
        This blending function is not restricted to two inputs like the basic
        (wikipedia) version. Any number works.
        
        and it brings the symetry to the front. 
        
        >>> assert A & B == B & A 
        >>> assert A & B == 1/(1/A+1/B)
        
        >>> abc=[A,B,C]
        >>> numpy.random.shuffle(abc)
        >>> assert A & B & C == paralell(*abc)
        >>> assert A & B & C == Mvar.blend(*abc)== Mvar.__and__(*abc)
        
        the proof that this is identical to the wikipedia definition of blend 
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
        
        This definition was developed to turn kalman blending into a standard 
        resistor-style 'paralell' operation
        
        The main idea is that only the scale matrix (eigenvalues) gets powered.
        (which is normal for diagonalizable matrixes), stretching the sheet at 
        an independant rate along each (perpendicular) eigenvector
        
        Because the scale matrix is a diagonal, powers on it are easy, 
        so this is not restricted to integer powers
        
        But the mean is also affected by the stretching. It's as if the usual 
        value of the mean is a "zero power mean" transformed by whatever is 
        the current value of the A.rotation.H*A.vectors matrix and if you change that 
        the mean changes with it..
        
        Most things you expect to work just work.
        
            >>> assert A**0== A**(-1)*A== A*A**(-1)== A/A        
            >>> assert (A**K1)*(A**K2)==A**(K1+K2)
            >>> assert A**K1/A**K2==A**(K1-K2)
        
        Zero power has some interesting properties: 
            
            The resulting ellipse is always a unit sphere, with the orientation 
            unchanged, but the mean is wherever it gets stretched to while we 
            transform the ellipse to a sphere
              
            >>> assert (A**0).scale==numpy.eye(A.mean.size)
            >>> assert (A**0).rotation== A.rotation
            >>> assert (A**0).mean==dot(
            ...     A.mean,A.rotation.H,A.scale**-1,A.rotation
            ... )
            
        derivation of multiplication:
        
            >>> assert A.vectors == dot(A.scale,A.rotation)
            >>> assert (A**K1).vectors == dot(numpy.diag(A.scale.diagonal()**K1),A.rotation)
            >>> assert (A**K1).vectors == dot(A.vectors,A.rotation.H,numpy.diag(A.scale.diagonal()**(K1-1)),A.rotation)
            >>> assert (A**K1).mean == dot(A.mean,A.rotation.H,numpy.diag(A.scale.diagonal()**(K1-1)),A.rotation)
            
            that's a matrix multiply.
            
            So all Mvars on the right,in a multiply, can just be converted to 
            matrix:
            
            >>> assert A*B==A*dot(B.rotation.H,B.scale,B.rotation)
        """
        rotation = self.rotation
        new_scale = Matrix(
            numpy.diag(numpy.diagonal(self.scale)**(power-1))
        )
            
        transform = dot(rotation.H,new_scale,rotation)
        
        return self*transform
        
    def __mul__(
        self,
        other,
        #this function is applied to the right operand, to simplify the types 
        #of the objects we have to deal with, and to convert any Mvars on 
        #the left into a rotation.H*scale*rotation matrix.
        rconvert=lambda 
            item,
            helper=lambda item:(
                Matrix(item) 
                if item.ndim else
                item
            )
        :(  
            Matrix(dot(item.rotation.H,item.scale,item.rotation)) if 

            isinstance(item,Mvar) else 
            helper(numpy.array(item))
        ),
        #this dict is used to dispatch multiplications based on the type of 
        #the right operand, after it has been passed through rconvert
        multipliers={
            (Matrix):(
                lambda self,matrix:Mvar.from_attr(
                    mean=dot(self.mean,matrix),
                    vectors=dot(self.scale,self.rotation,matrix),
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
            rotation.H*scale*rotation matrix and the multiplication is 
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
            >>> assert (A*B).affine==(A*dot(B.rotation.H,B.vectors)).affine
            >>> assert (A*B).mean==dot(A.mean,B.rotation.H,B.scale,B.rotation)
            >>> assert A*(B**2) == A*(B.cov)
            
            Note that the result does not depend on the mean of the 
            second mvar(!) (really any mvar after the leftmost mvar or matrix)

        Mvar*constant == constant*Mvar
            Matrix multiplication and scalar multiplication behave differently 
            from eachother.  
            
            For this to be a properly defined vector space scalar 
            multiplication must fit with addition, and addition here is 
            defined so it can be used in the kalman noise addition step so: 
            
            >>> assert (A+A).vectors==(2*A).vectors
            >>> assert (A+A).vectors==(2**0.5)*A.vectors
            
            >>> assert (A+A).mean==(2*A).mean
            >>> assert (A+A).mean==2*A.mean
            
            >>> assert (A*K1).vectors==(K1**0.5)*A.vectors
            >>> assert (A*K1).mean==K1*A.mean
            
            >>> assert sum(itertools.repeat(A,N-1),A) == A*(N)
            
            >>> assert (A*K1).cov==A.cov*K1
            
            be careful with negative constants because you will end up with 
            imaginary numbers in you vectors matrix, (and lime in your coconut) as 
            a direct result of:            
            
            assert (A*K1).vectors==(K1**0.5)*A.vectors
            assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
            
            if you want to scale the distribution linearily with the mean
            then use matrix multiplication
        
        Mvar*matrix
        
            matrix multiplication transforms the mean and ellipse of the 
            distribution. Defined this way to work with the kalman state 
            update step.
            
            simple scale is like this
            >>> assert (A*(numpy.eye(A.mean.size)*K1)).vectors==A.vectors*K1
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
        #or be converted to a rotation.H*scale*rotation matrix for matrix 
        #multiplication
        convert=lambda
            other,
            self,
            helper=lambda other,self: (
                Matrix(other) if 
                other.ndim else
                other
                ,
                Matrix(dot(self.rotation.H,self.scale,self.rotation)) if 
                other.ndim else
                self
            )
        :helper(numpy.array(other),self)
        ,
        #dispatch the multiplication based on the type of the left operand
        multipliers={
            #if the left operand is a matrix, the mvar has been converted to
            #to a matrix -> use matrix multiply
            (Matrix):dot,
            #if the left operand is a constant use scalar multiply
            (numpy.ndarray):(
                lambda constant,self:Mvar.from_cov(
                    mean= constant*self.mean,
                    cov = constant*self.cov,
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
            >>> assert A*B == A*dot(B.rotation.H,B.vectors) 
            
            the second Mvar is automatically converted to a matrix, and the 
            result is handled by matrix multiply
            
            again note that the result does not depend on the mean of the 
            second mvar(!)
        
        martix*Mvar
            >>> assert M*A==dot(M,A.rotation.H,A.scale,A.rotation)

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
        
        >>> assert (A+A).vectors==(2*A).vectors
        >>> assert (A+A).vectors==(2**0.5)*A.vectors
        
        >>> assert (A+A).mean==(2*A).mean
        >>> assert (A+A).mean==2*A.mean
        
        >>> assert (A+B).mean==A.mean+B.mean
        >>> assert (A+B).cov==A.cov+B.cov
        
        watch out subtraction is the inverse of addition 
         
            >>> assert (A-B)+B == A
            >>> assert (A-B).mean== A.mean- B.mean
            >>> assert (A-B).cov== A.cov - B.cov
            
        if you want something that acts like rand()-rand() use:
            
            >>> assert (A+B*(-1*numpy.eye(A.mean.size))).mean== A.mean - B.mean
            >>> assert (A+B*(-1*numpy.eye(A.mean.size))).cov== A.cov + B.cov

        __sub__ also fits with __neg__, __add__, and scalar multiplication.
        
            >>> assert B+(-A) == B+(-1)*A == B-A
            
            but watchout you'll end up with complex... everything?
        """
        return Mvar.from_cov(
            mean= (self.mean+other.mean),
            cov = (self.cov + other.cov),
        )
        
    ################# Non-Math python internals
    def __iter__(self):
        pass
        
    def __repr__(self):
        return '\n'.join([
            'Mvar.from_attr(',
            '    mean=',8*' '+self.mean.__repr__().replace('\n','\n'+8*' ')+',',
            '    scale=',8*' '+self.scale.__repr__().replace('\n','\n'+8*' ')+',',
            '    vectors=',8*' '+self.rotation.__repr__().replace('\n','\n'+8*' ')+',',
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
            #and rotation angle pulled from the rotation matrix
            angle=numpy.rad2deg(
                numpy.angle(astype(
                    self.rotation,
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
    yk=M.mean.T-P.mean.T
    Sk=P.cov+M.cov
    Kk=dot(P.cov,Matrix(Sk).I)
    
    return Mvar.from_cov(
        mean=(P.mean.T+dot(Kk,yk)).T,
        cov=dot((numpy.eye(P.mean.size)-Kk),P.cov)
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

    
def issquare(A):
    shape=A.shape
    return A.ndim==2 and shape[0] == shape[1]

def isrotation(A):
    R=Matrix(A)
    return (R*R.H == eye(R.shape[0])).all()

def isdiag(A):
    shape=A.shape
    return A.ndim==2 and ((A != 0) == numpy.eye(shape[0],shape[1])).all()


if __name__=="__main__":
    import doctest
    #create random test objects
    A=Mvar.from_attr(mean=10*numpy.random.randn(1,2),vectors=10*astype(numpy.random.randn(2,2,2),complex))
    B=Mvar.from_cov(mean=10*numpy.random.randn(1,2),cov=(lambda x:dot(x,x.H))(10*Matrix(astype(numpy.random.randn(2,2,2),complex))))
    C=Mvar.from_data(numpy.dot(astype(numpy.random.randn(50,2,2),complex),10*astype(numpy.random.randn(2,2,2),complex)))
   
    
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

    

    
    
