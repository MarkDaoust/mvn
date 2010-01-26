##imports
#internals
from __future__ import division

#builtins
import itertools
from itertools import izip as zip

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
from helpers import autostack,diagstack,astype,paralell,close,rotation2d


class Mvar(object):
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
        
    normally these things are handled with mean and covariance, but I find
    mean,scale,vectors to be more useful, so that is how the data is actually, 
    managed, but other useful info in accesable through virtual attributes
    (properties).
    
    This system make compression much easier (and more useful)
    
    actual attributes:
        mean
            mean of the distribution
        scale
            the eiggenvalue asociated with each eigenvector
        vectors
            unit eigenvectors, as rows
    virtual attributes:    
        scaled
            scaled eigenvectors, as rows
        cov
            covariance matrix
        affine
            autostack([
                [self.scaled,numpy.zeros]
                [self.mean  ,          1]
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
    to do a matrix multiplies on it. this is for two reasons: 

        1) inplace operators work nicely (Mvar on the left)

        2) The Mvar is the only object that knows how to do operations on 
        itself, might as well go straight to it instead of passing around 
        'NotImplemented's 
        
    No work has been done to make things fast, because until they work at all
    and speed actually is a problem it's not worth working on. 
    """
    
    ############## Creation
    def __init__(self,
        mean = numpy.zeros,
        vectors = numpy.zeros, 
        scale = numpy.ones,
        do_square=False,
        do_compress=True,
    ):
        """
        create the Mvar directly from it's attributes.
        
        uses autostack to automatically size any callables passed to the 
        attributes, it's rarely useful, it's just convienient.
        
            autostack([
                [scale,vectors],
                [    1,   mean],
            ])
        
        mean
            defaults to numpy.zeros, sounld be a row vector
            
        vectors
            defaults to numpy.zeros, row vectors
            does not need to be orthogonal, or unit vectors
            
        scale
            defaults to numpy.ones
            
        do_square 
            calls self.do_square() on the result if true. To set the vectors 
            to orthogonal and unit length, do_square automatically calls 
            do_compress, 
            
        do_compress
            calls self.do_compress() on the result if true. To clear out any 
            low valued vectors uses the same defaults as numpy.allclose()
        """
        scale=(
            scale if callable(scale) else 
            numpy.matrix(numpy.array(scale).flat).T
        )
        
        #use autostack to determine unknown sizes, and matrix everything
        stack=autostack([
            [scale,vectors],
            [    1,   mean],
        ])
        
        #and unpack the stack into the object's parameters 
        self.mean = stack[-1,1:]
        self.scale = stack[:-1,0]
        self.vectors = stack[:-1,1:]
        
        if do_square:
            assert do_compress
        
        if do_square:
            self.do_square()
        elif do_compress:
            self.do_compress()
    
    def do_square(self,**kwargs):
        """
        this is NOT x**2 it is to set the vectors to perpendicular
        **kwargs is just passed on to do_compress
        """
        V=self.vectors
        if not numpy.allclose(V*V.T,numpy.eye(V.shape[0])):
            (scale2,self.vectors) = numpy.linalg.eigh(self.V.T*V)
            self.scale*=scale2**(0.5)
        
        self.do_compress(**kwargs)
        
    def do_compress(self,rtol=1e-5,atol=1e-8):
        """
        drop and vector/scale pairs which are under the tolerence limits
        the defaults match numpy's for 'allclose'
        """
        #convert the scale to a column vector
        scale=numpy.matrix(numpy.array(self.scale).flat).T
        #get the vectors
        vectors=numpy.array(self.vectors)
        
        stack=numpy.hstack([scale,vectors])

        stack=stack[numpy.argsort(scale.flat),:]
        
        C=~numpy.array(close(stack[:,0],rtol=rtol,atol=atol)).squeeze()
        #drop the scale/vectors where the scale is close to zero
        stack=stack[C,:]
        #unstack them
        self.scale = stack[:,0]
        self.vectors=stack[:,1:] 
        ############## alternate creation methods
    @staticmethod
    def from_cov(cov,**kwargs):
        """
        everything in kwargs is passed directly to the constructor
        don't set do_square to true, they will automatically be orthogonal when 
        pulled out of the covariance
        """
        #get the scale and rotation matracies
        scale,vectors = numpy.linalg.eigh(cov)
        
        return Mvar(
            vectors=vectors,
            #square root the scales
            scale=scale**0.5,
            **kwargs
        )
    
    @staticmethod
    def from_data(data, bias=0, rowvar=0, **kwargs):
        """
        the kwargs are just passed to the basic constructor.
        
        create an Mvar with the same mean and covariance as the supplied data
        with each row being a sample and each column being a dimenson
        
        remember numpy's default covariance calculation divides by (n-1) not 
        (n) set bias = 1 to use N,
        
        my default for rowvar is the opposite of numpy's
        """
        #if variables are along rows, switch to colums
        if rowvar:
            data=data.T
            
        #convert the data to a matrix 
        data=numpy.matrix(data)
        #create the mvar from the mean and covariance of the data
        return Mvar.from_cov(
            cov = numpy.cov(data.T, bias=bias,rowvar=0),
            mean= numpy.mean(data,axis=0),
            **kwargs
        )

    def copy(self,other):
        self.__dict__=other.__dict__.copy()
        
    ########## Utilities
    @staticmethod
    def stack(*mvars,**kwargs):
        """
        it's a static method to make it clear that it's not happening in place
        Stack two Mvars together, equivalent to hstacking the vectors, and 
        diagstacking the covariance matrixes
        
        yes it works but be careful. Don't use this for reconnecting 
        something you calculated from an Mvar, back to the same Mvar it was 
        calculated from, you'll loose all the cross corelations. 
        If you're trying to do that use a better matrix multiply. 
        """
        #no 'refresh' is necessary here because the vectors are in entierly 
        #different dimensions
        return Mvar(
            #stack the means horizontally
            mean=numpy.hstack([mvar.mean for mvar in mvars]),
            #stack the vector packets diagonally
            vectors=diagstack([mvar.vectors for mvar in mvars]),
            scale=numpy.hstack([numpy.matrix(mvar.scale) for scale in mvars]),
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
            numpy.matrix(numpy.random.randn(n,self.mean.size)),
            numpy.matrix(numpy.ones([n,1])),
        ])
        
        transform=numpy.vstack([
            self.scaled,
            self.mean,
        ])

        return data*transform
    
    ############ get methods for all the properties

    def get_cov(self):
        """
        get the covariance matrix used by the object
        
        >>>assert A.cov == A.scaled.T*A.scaled 
        >>>assert A.cov == A.vectors.T*A.scale.T*A.scale*A.vectors 
        >>>assert A.cov == A.vectors.T*A.scale**2*A.vectors
        """
        scaled=self.scaled
        return scaled.T*scaled
    
    def get_scaled(self):
        """
        get the matrix of scaled eigenvectors (as rows)

        >>>assert A.scaled = numpy.diagflat(A.scale)*A.vectors 
        """
        return numpy.diagflat(self.scale)*self.vectors 

    ############ Properties
    #maybe later I'll add in some of those from functions
    cov   = property(fget=get_cov)
    scaled   = property(fget=get_scaled)
    
    ############ Math

    #### logical operations
    def __eq__(self,other):
        """
        A==
        compares the means and covariances or the distributions
        """
        return (self.mean==other.mean).all() and (self.cov == other.cov).all()
        
    def __and__(*mvars):
        """
        A & 
        
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
        
        >>>assert A & B == B & A 
        >>>assert A & B == 1/(1/A+1/B)
        >>>assert A & B & C == Paralell(B,C,A)
        >>>assert A & B & C == Mvar.__and__(B,A,C)
        
        the proof that this is identical to the wikipedia definition of blend 
        is a little too involved to write here. Just try it (see the wiki 
        function below)
        
        >>>assert A & B == wiki(A,B)
        """
        return paralell(mvars)

    def __iand__(*mvars):
        """
        A&=
        
        see documentation of __and__
        """
        self.copy(paralell(mvars))

    ## operators
    def __pow__(self,power):
        """
        A**
        
        This definition was developed to turn kalman blending into a standard 
        resistor-style 'paralell' operation
        
        The main idea is that only the scale matrix (eigenvalues) gets powered.
        (which is normal for diagonalizable matrixes), stretching the sheet at 
        an independant rate along each (perpendicular) eigenvector
        
        Because the scale matrix is a diagonal, powers on it are easy, 
        so this is not restricted to integer powers
        
        But the mean is also affected by the stretching. It's as if the usual 
        value of the mean is a "zero power mean" transformed by whatever is 
        the current value of the A.vectors.T*A.scaled matrix and if you change that 
        the mean changes with it..
        
        Most things you expect to work just work.
        
            >>>assert A**0== A**(-1)*A== A*A**(-1)== A/A        
            >>>assert (A**K1)*(A**K2)=A**(K1+K2)
            >>>assert A**K1/A**K2=A**(K1-K2)
        
        Zero power has some interesting properties: 
            
            The resulting ellipse is always a unit sphere, with the orientation 
            unchanged, but the mean is wherever it gets stretched to while we 
            transform the ellipse to a sphere
              
            >>>assert (A**0).scale== eye
            >>>assert (A**0).vectors== A.vectors
            >>>assert (A**0).mean == A.mean*A.vectors.T*A.scale**-1*A.vectors
            
        derivation of multiplication:
        
            >>>assert A.scaled== A.scale*A.vectors
            >>>assert (A**K).scaled== (A.scale**K)*A.vectors
            >>>assert (A**K).scaled== A.scaled* A.vectors.T*A.scale**(K-1)*A.vectors
            >>>assert (A**K).mean== A.mean* A.vectors.T*A.scale**(K-1)*A.vectors
            
            that's a matrix multiply.
            
            So all Mvars on the right,in a multiply, can just be converted to 
            matrix:
            
            >>>assert A*B==A*(B.vectors.T*B.scale*B.vectors)
        """
        vectors = self.vectors
        scale = self.scale
        
        transform = (
            vectors.T*
            numpy.matrix(numpy.diag(scale**(power-1)))*
            vectors
        )
        
        return self*transform
    
    def __ipow__(self,power):
        """
        A**=
        """
        self.copy(self**power)
        
    def __mul__(self,other):
        """
        A*
        
        coercion notes:
            All non Mvar imputs will be converted to numpy arrays, then 
            treated as constants if zero dimensional, or matrixes otherwise 
            
            Mvar always beats constant. Between Mvar and numpy.matrix the left 
            operand wins 
            
            >>>assert isinstance(A*B,Mvar)
            >>>assert isinstance(A*M,Mvar)
            >>>assert isinstance(M*A,numpy.Matrix) 
            >>>assert isinstance(A*K,Mvar)
            >>>assert isinstance(K*A,Mvar)
            
            whenever an mvar is found on the right it is converted to a 
            vectors.T*scale*vectors matrix and the multiplication is 
            re-called.
            
        general properties:
            
            constants still commute            
            >>>assert K*A*M == A*K*M == A*M*K
            
            but the asociative property is lost if you mix constants and 
            matrixes (but I think it's ok if you only have 1 of the two types?)
            
            >>>assert (A*2)*M == A*(4*M)
            
            ????
            asociative if only mvars and matrixes?
            ????
            still distributive?
            
        Mvar*Mvar
            multiplying two Mvars together is defined to fit with power
            
            >>>assert A*A==A**2
            >>>assert (A*B).affine=A.affine*B.vectors.T*B.scaled
            >>>assert (A*B).scaled == A.scaled*B.vectors.T*B.scale*B.vectors
            >>>assert (A*B).mean == A.mean*B.vectors.T*B.scale*B.vectors
            
            Note that the result does not depend on the mean of the 
            second mvar(!) (really any mvar after the leftmost mvar or matrix)

        Mvar*constant == constant*Mvar
            Matrix multiplication and scalar multiplication behave differently 
            from eachother.  
            
            For this to be a properly defined vector space scalar 
            multiplication must fit with addition, and addition here is 
            defined so it can be used in the kalman noise addition step so: 
            
            >>>assert ((A+A).scaled == (2*A).scaled).all()
            >>>assert ((A+A) == sqrt(2)*A.scaled).all()
            >>>assert ((A+A).mean == (2*A).mean).all()
            >>>assert ((A+A).mean == 2*A.mean).all()
            
            >>>assert ((A*K).scaled == sqrt(K)*A.scaled).all()
            >>>assert ((A*K).mean == K*A.mean).all()
            
            >>>assert sum(itertools.repeat(A,K-1),A) == A*(K) == (K)*A 
            
            >>>assert ((A*K).cov == A.cov*K).all()
            
            be careful with negative constants because you will end up with 
            imaginary numbers in you scaled matrix, (and lime in your coconut) as 
            a direct result of:            
            
            assert ((A*K).scaled == sqrt(K)*A.scaled).all()
            assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
            
            if you want to scale the distribution linearily with the mean
            then use matrix multiplication
        
        Mvar*matrix
        
            matrix multiplication transforms the mean and ellipse of the 
            distribution. Defined this way to work with the kalman state 
            update step.
            
            simple scale is like this
            >>>assert ((A(*eye*K)).scaled == A.scaled*K).all()
            >>>assert ((A(*eye*K)).mean == A.mean*K).all()
            
            or more generally
            >>>assert (A*M).cov == M.T*A.cov*M
            >>>assert (A*M).mean == A.mean*M
            
            matrix multiplication is implemented as follows
            
            assert A*M == Mvar(A.affine*diagstack([M,1])).refresh()
            
            the refresh() here is necessary to ensure that the rotation matrix
            stored in the object stays well behaved. 
        """
        return multiply(self,other)
    
    def __rmul__(self,other):
        """
        *A
        
        multiplication order for constants doesn't matter
        
        >>>assert k*A == A*k
        
        but it matters a lot for Matrix/Mvar multiplication
        
        >>>assert isinstance(A*T,Mvar)
        >>>assert isinstance(T*A,numpy.matrix)
        
        be careful with right multiplying:
        Because power must fit with multiplication
        
        assert A*A==A**2
        
        The most obvious way to treat right multiplication by a matrix is to 
        do exactly the same thing. So because of the definition of Mvar*Mvar
        (below)
        
        Mvar*Mvar
            multiplying two Mvars together fits with the definition of power
            
            assert prod(itertools.repeat(A,N)) == A**N
            assert A*B == A*(B.vectors.T*B.scaled) 
            
            the second Mvar is automatically converted to a matrix, and the 
            result is handled by matrix multiply
            
            again note that the result does not depend on the mean of the 
            second mvar(!)

        for consistancy when right multiplied, an Mvar is always converted to 
        the A.vectors.T*A.scaled matrix, and Matrix multiplication follows 
        automatically, and yields a matrix, not an Mvar.
        
        the one place this automatic conversion is not applied is when 
        right multiplying by a constant so: 
        
        martix*Mvar
            assert T*A == T*(A.vectors.T*A.scale*A.vectors)

        scalar multiplication however is not changed.
        
        assert Mvar*constant == constant*Mvar
        """
        return multiply(other,self)
    
    def __imul__(self,other):
        """
        A*=
        
        This is why I have things set up for left multply, it's 
        so that __imul__ works.
        """
        self.affine=multiply(self,other).affine
    
    def __div__(self,other):
        """
        A/
        
        see __mul__ and __pow__
        it would be immoral to overload power and multiply but not divide 
        >>>assert A/B == A*(B**(-1))
        >>>assert A/M == A*(M**(-1))
        >>>assert A/K == A*(K**(-1))
        """
        return multiply(self,other**(-1))
        
    def __rdiv__(self,other):
        """
        /A
        
        see __rmul__ and __pow__
        >>>assert K/A == K*(A**(-1))
        >>>assert M/A == M*(A**(-1))
        """
        return multiply(other,self**(-1))
        
    def __idiv__(self,other):
        """
        A/=
        
        see __mul__ and __pow__
        >>self.affine=(self*other**(-1)).affine
        """
        self.affine=multiply(self,other**(-1))
        
    def __add__(self,other):
        """
        A+
        
        When using addition keep in mind that rand()+rand() is not like scaling 
        one random number by 2, it adds together two random numbers.

        The add here is like rand()+rand()
        
        Addition is defined this way so it can be used directly in the kalman 
        noise addition step
        
        so if you want simple scale use matrix multiplication like rand()*(2*eye)
        
        scalar multiplication however fits with addition:
        
        >>>assert (A+A).scaled == (2*A).scaled == sqrt(2)*A.scaled
        >>>assert (A+A).mean == (2*A).mean == 2*A.mean

        >>>assert (A+B).mean== A.mean+B.mean
        >>>assert (A+B).cov == A.cov+B.cov

        it also works with __neg__, __sub__, and scalar multiplication.
        
        assert B+(-A) == B+(-1)*A == B-A and (B-A)+A=B
        
        but watchout you'll end up with complex eigenvalues in your scaled 
        matrix's
        """
        try:
            return Mvar.from_cov(
                mean= (self.mean+other.mean),
                cov = (self.cov + other.cov),
            )
        except AttributeError:
            return NotImplemented

    def __radd__(self,other):
        """
        +A
        """
        self.copy(self+other)

    def __iadd__(self,other):
        """
        A+=
        """
        self.affine = (self+other).affine

    def __sub__(self,other):
        """
        A-
        
        watch out subtraction is the inverse of addition 
         
            assert (A-B)+B == A
            assert (A-B).mean ==A.mean- B.mean
            assert (A-B).cov ==A.cov - B.cov
            
        if you want something that acts like rand()-rand() use:
            
            assert (A+B*(-1*eye)).mean == A.mean - B.mean
            assert (A+B*(-1*eye)).cov == A.cov + B.cov

        __sub__ also fits with __neg__, __add__, and scalar multiplication.
        
        assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
        """
        try:
            return Mvar.from_mean_cov(
                mean= (self.mean-other.mean),
                cov = (self.cov - other.cov),
            )
        except AttributError:
            return NotImplemented

    def __rsub__(self,other):
        return self+other
    
    def __isub__(self, other):
        """
        A-=
        """
        self.copy(self-other)

    def __neg__(self):
        """
        -A
        
        it would be silly to overload __sub__ without overloading __neg__
        
        assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
        """
        return (-1)*self
    
    ################# Non-Math python internals
    def __str__(self):
        return '\n'.join([
            'Mvar(',
            '    mean=',8*' '+str(self.mean).replace('\n','\n'+8*' ')+',',
            '    scale=',8*' '+str(self.scale).replace('\n','\n'+8*' ')+',',
            '    vectors=',8*' '+str(self.vectors).replace('\n','\n'+8*' ')+',',
            ')',
        ])
    
    def __repr__(self):
        return '\n'.join([
            'Mvar(',
            '    mean=',8*' '+self.mean.__repr__().replace('\n','\n'+8*' ')+',',
            '    scale=',8*' '+self.scale.__repr__().replace('\n','\n'+8*' ')+',',
            '    vectors=',8*' '+self.vectors.__repr__().replace('\n','\n'+8*' ')+',',
            ')',
        ])

    ################ Art
    def get_patch(self,nstd=4,**kwargs):
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
            xy=tuple(self.mean.flat),
            #matching width and height
            width=width, height=height,
            #and rotation angle pulled from the rotation matrix
            angle=numpy.rad2deg(
                numpy.angle(astype(
                    self.vectors,
                    complex,
                )).flat[0]
            ),
            #while transmitting any kwargs.
            **kwargs
        )


## extras    

def wiki(P,M):
    """
    Direct implementation of the wikipedia blending algorythm
    
    The quickest way to prove it's equivalent is by:
        >>>ab=numpy.array([A,B],ndmin=2,dytpe=object)
        >>>assert A & B == numpy.dot(ab,(ab.T)**(-2))**(-1)
    """
    yk=M.mean.T-P.mean.T
    Sk=P.cov+M.cov
    Kk=P.cov*(Sk**-1)
    
    return Mvar.from_mean_cov(
        (P.mean.T+Kk*yk).T,
        (numpy.eye(P.mean.size)-Kk)*P.cov
    )

def isplit(sequence,fkey=bool): 
    """
    return a defaultdict (where the default is an empty list), 
    where every value is a sub iterator produced from the sequence
    where items are sent to iterators based on the value of fkey(item).
    
    >>>isodd = isplit([xrange(1,7),lambda item:bool(item%2))
    >>>
    >>>list(isodd[True])
    [1,3,5]
    >>>list(isodd[False])
    [2,4,6]
    
    which gives the same results as
    
    >>>X=xrange(1,7)
    >>>[item for item in X if bool(item%2)]
    [1,3,5]
    >>>[item for item in X if not bool(item%2)]
    [2,4,6]
    
    or you could make a mess of maps and filter
    but this is so smooth,and really shortens things 
    when dealing with a lot of keys 
    
    >>>bytype = isplit([1,'a',True,"abc",5,7,False],type)
    >>>
    >>>list(bytype[int])
    [1,5,7]
    >>>list(bytype[str])
    ['a','abc']
    >>>list(bytype[bool])
    [True,False]
    >>>list(bytype[dict])
    []
    """
    
    return collections.defaultdict(
        list,
        itertools.groupby(sequence,fkey),
    )
    
def product(*args):
    """
    like sum, but with multiply
    not like numpy product which works element-wise.
    """
    return reduce(
        function=operator.mul,
        sequence=args,
        initial=1
    )
    
#remember when reading this that default values for function arguments are 
#only evaluated once. 
#I'm only defining this outside of the class because you can't point to the 
#class until the class is done being defined...

def multiply(
    self,
    other,
    rconvert=lambda(item): (
        Mvar if isinstance(item,Mvar) else 
        numpy.matix(item) if numpy.array(item).ndim else
        numpy.ndarray(item)
    ),
    multipliers=(
        lambda scalarmul,rmul,lmul:{
            (Mvar,Mvar):rmul,
            (numpy.matrix, Mvar): rmul,
            (Mvar, numpy.matrix): lmul,
            (Mvar, numpy.ndarray): scalarmul,
            (numpy.ndarray, Mvar): scalarmul,
        }
    )(
        scalarmul=lambda self,constant: Mvar.from_mean_cov(
            mean= first.mean*constant,
            cov = first.cov*constant,
        ),
        rmul=lambda other,self:(
            other*self.vectors.T*self.scaled
        ),
        lmul=lambda self,matrix: Mvar(
            self.affine*diagstack([mat, 1])
        ).refresh(),
    ),
):
    other = rconvert(other)
    return multipliers[
        (type(self),type(other))
    ](self,other)
    
def issquare(matrix):
    shape=numpy.matrix(matrix).shape
    return shape[0] == shape[1]

def isflat(matrix):
    shape=numpy.matrix(matrix).shape
    return min(shape)==1

def isrotate(matrix):
    R=numpy.martix(matrix)
    return (R*R.T == eye(R.shape[0])).all()


"""
>>>assert A.vectors*A.vectors.T == eye
>>>assert A.cov = A.vectors*A.scale**2*A.vectors.T
"""
