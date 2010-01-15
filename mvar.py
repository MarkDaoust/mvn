import itertools
import numpy
import operator


try:
    from matplotlib.patches import Ellipse
except ImportError:
    def Ellipse(*args,**kwargs):
        """
        Unable to find matplotlib.patches.Ellipse
        I know the zen of python says "Errors should never be passed silently"
        but this is a side feature, and ifyou wantit you'll find out soon 
        enough that it's not working 
        """
        raise ImportError(
            "Ellipse is required, from matplotlib, to get a patch"
        )

class Mvar(object):
    """
    Multivariate normal distributions packaged to act like a vector.
        
    This is done with kalman filtering in mind, but is good for anything where 
        you need to track linked uncertianties across multiple variables.
    
    basic math (+,-,*,/,**, and) has been overloaded to work 'normally', but 
    there are *a lot* of surprising features of the math these things produce, 
    so watchout. The most surprising one is that multilpy is not asociative if 
    you mix constants and matrixes!
        
    The data is stored as an affine transformation, that is one large matrix 
    containing the mean and standard-deviation matrixes.
    
        >>>assert A.affine == autostack([
        ...    [A.std,numpy.zeros],
        ...    [A.mean,1],
        ...])
        
        The "standard deviation" matrix is a scaled eigenvector matrix
        where each row is a scaled vector.
        
        It will make compression easier (in the future when I get to it) but 
        certian other things harder.   
        
        you can think of *everything* as being embeded on a sheet in a space 
        with 1 extra dimension with the sheet located 1 unit "up" along the 
        extra dimension.
        
        with this setup some math operations are greatly simplified.
    
    the from* functions all create new instances from varous 
    common constructs.
        
    the get* functions all grab useful things out of the structure, 
    all have equivalent Properties linked to them
    
    the inplace operators work but, unlike in many classes, do not 
    speed up any operations.
    
    the mean is stored as a row vector, so make sure to modify your transforms 
    apropriately and have the Mvar on left the when attemptingto do a matrix 
    transform on it. this is for two reasons: 

        1) inplace operators work nicely (Mvar on the left)

        2) The Mvar is the only object that knows how to do operations on 
        itself, might as well go straight to it instead of passing around 
        NotImplemented's 
        
    No work has been done to make things fast, because until they work 
    and the speed actually is a problem it's not worth working on. 
    """
    
    ############## Creation
    def __init__(self,affine):
        """
        create an instance directly from an affine transform.
        the transform should be of the format:
        
        affine = autostack([
            [std,numpy.zeros],
            [mean,1],
        ])
            
        the affine transform is the only real attribute in these things. 
        everything else is calculated on the fly
        
        standard deviation, 'std' is not a common term when dealing with these,
        let me explain:
        
        'scale' is a diagonal matrix of eigenvalues
        
        >>>assert A.scale == numpy.matrix(numpy.diag(numpy.sqrt(
        ...    numpy.linalg.eigenvalues(A.cov)
        ...)))
        
        'rotate' is a matrix of row-eigenvectors
        
        >>>assert A.rotate == numpy.linalg.eigh(A.cov)[1]
        
        >>>assert A.std==A.scale*A.rotate
        >>>assert A.std**(-1)==A.rotate.T*A.scale**(-1)
        
        >>>assert A.rotate.T == A.rotate**(-1)
        
        >>>A.std.T*A.std == A.rotate.T*A.scale*A.scale*A.rotate == A.cov
        >>>A.std*A.std.T == A.scale*A.rotate*A.rotate.T*A.scale ==A.scale**2
        
        (A.std*A.std.T behaves like a dot product, A.std.T*A.std behaves like 
        an outer product, a covariance matrix is the average outerproduct of 
        the distribution.... interesting, I wonder whatit means)
        """
        self.affine=affine
    
    def refresh(self):
        """
        transforms knock the eigenvectors out of orthogonality, 
        this function just realigns them. It is done in place intentionally. 
        it also returnsthe self... is that going to cause problems?  
        """
        self.affine = Mvar.from_mean_cov(self.mean, self.cov).affine
        return self
    
    ############## alternate creation methods
    #maybe eventually I'll link these to the properties 
    #to make them all writeable
    
    @staticmethod
    def from_mean_std(mean=None,std=None):
        """
        if either parameter is missing it gets filled with zeros,
        at least one must be supplied so the size is known.
        """
        assert (mean is not None) or (std is not None)
        
        mean=numpy.matrix(mean) if mean is not None else None
        std=numpy.matrix(std) if mean is not None else None
        
        #if only one input is supplied, assume the other is all zeros
        std = zeros((mean.size,mean.size)) if std is None else std
        mean = zeros((1,std.shape[0])) if mean is None else mean
        
        return Mvar(autostack([
            [ std, numpy.zeros([std.shape[0],1])],
            [mean, 1],
        ]))
            
    
    @staticmethod
    def from_mean_cov(mean = None,cov = None):
        """
        if either one is missing it gets filled with zeros,
        at least one must be supplied so the size can be calculated.
        """
        #convert inputs to matrixes 
        cov = None if cov is None else numpy.matrix(cov) 
        mean = None if mean is None else numpy.matrix(mean)
        
        #if only one input is supplied, assume the other is all zeros
        cov = numpy.zeros((mean.size,mean.size)) if cov is None else cov
        mean = numpy.zeros((1,cov.shape[0])) if mean is None else mean
        
        scale,rotation = numpy.linalg.eigh(cov)
        
        #get the square root of the scales 
        scale = scale**0.5
        
        #create the affine transform, and from it the Mvar
        return Mvar.from_roto_scale(
            rotation = rotation,
            scale = scale,
            mean = mean,
        )
    
    @staticmethod
    def from_data(data, bias=0):
        """
        create an Mvar with the same mean and covariance as the supplied data
        with each row being a sample and each column being a dimenson
        
        remember numpy's default covariance calculation divides by (n-1) not 
        (n) set bias = 1 to use N
        """
        #convert the data to a matrix 
        data=numpy.matrix(data)
        
        #create the mvar from the mean and covariance of the data
        return Mvar.from_mean_cov(numpy.mean(data), numpy.cov(data, bias=bias))
    
    @staticmethod
    def from_roto_scale(rotation,scale,mean=None):
        """
        Rotation can be either a matrix or, if the mean and scale are 
        two dimensional, it can be the rotation angle (radians) and the 
        rotation matrix will be created automatically.
        
        Scale can be a vector, or a diagonal matrix. 
        Each element in the scale matrix is the scale 
        along the corresponding vector in the rotation matrix 
        """
        #convert everything to matrixes
        rotation,scale,mean=(
            numpy.matrix(data) if data is not None else None 
            for data in (rotation,scale,mean)
        )
        
        #if the scale matrix is not square
        if scale.shape[0] != scale.shape[1]:
            #convert it to a diagonal matrix
            scale=numpy.matrix(numpy.diagflat(numpy.array(scale)))
        
        #if the rotatin matrix is a scalar, and we're working in 2 dimensions
        if rotation.size==1 and scale.shape==(2,2):
            # then rotation is the rotation angle, 
            #create the apropriate rotation matrix
            rotation=autostack([
                [ numpy.cos(rotation),numpy.sin(rotation)],
                [-numpy.sin(rotation),numpy.cos(rotation)],
            ])
        
        #if they didn't give a mean create zeros of the correct shape
        mean = numpy.zeros((1,scale.shape[0])) if mean is None else mean
        
        #create the Mvar
        return Mvar(autostack([
            [ scale*rotation, numpy.zeros([scale.shape[0],1])],
            [mean, 1],
        ]))

    @staticmethod
    def stack(*mvars):
        """
        Stack two Mvars together, equivalent to hstacking the vectors
        
        yes it works but be careful. Don't use this for reconnecting 
        something you calculated from an Mvar, back to the same Mvar it was 
        calculated from, you'll loose all the cross corelations. 
        If you're trying to do that use a better matrix multiply. 
        """
        #no 'refresh' is necessary here because the vectors are in entierly 
        #different dimensions
        return Mvar.from_mean_std(
            #stack the vectors horizontally
            numpy.hstack([mvar.mean for mvar in mvars]),
            #stack the covariances diagonally
            diagstack([mvar.std for mvar in mvars]),
        )
    
    ############ Eigen-stuf
    def get_rotate(self):
        """
        get the rotation matrix used in the object
        aka a matrix of normalized eigenvectors as rows
        
        >>>assert A.rotate*A.rotate.T == eye
        >>>assert A.cov = A.rotate*A.scale**2*A.rotate.T
        """
        return autostack([
            vector/numpy.sqrt(vector*vector.T) 
            for vector in self.std
        ])
        

    def get_scale(self):
        """
        get the scale matrix used in the object
        aka a diagonal matrix of eigenvalues (
        
        >>>assert A.std == A.scale*A.rotate
        >>>assert A.scale**2 == A.std.T*A.std
        >>>assert A.cov == A.rotate*A.scale**2*A.rotate
        """
        return numpy.matrix(numpy.diag(numpy.sqrt(numpy.diag(
            #all you're left with is the square of the scales on the diagonal
            self.std*self.std.T
        ))))


    def get_cov(self):
        """
        get the covariance matrix used by the object
        
        >>>assert A.cov == A.std.T*A.std 
        >>>assert A.cov == A.rotate.T*A.scale.T*A.scale*A.rotate 
        >>>assert A.cov == A.rotate.T*A.scale**2*A.rotate
        """
        return self.std.T*self.std
    
    def get_mean(self):
        """
        get the mean of the distribution (row vector)
        
        >>>assert Mvar.from_data(data).mean == numpy.mean(data)
        """
        return self.affine[-1,:-1]
    
    def get_std(self):
        """
        get the standard deviation of the distribution,
        aka matrix of scaled eigenvectors (as rows)

        >>>assert A.std = A.scale*A.rotate 
        """
        return self.affine[:-1,:-1]
    
    ############ Properties
    #maybe later I'll add in some of those from functions
    rotate= property(fget=get_rotate)
    scale = property(fget=get_scale)
    cov   = property(fget=get_cov)
    mean  = property(fget=get_mean)
    std   = property(fget=get_std)
    
    ############ Math
    def sample(self,n=1):
        """
        take samples from the distribution
        n is the number of samples, the default is 1
        each sample is a row vector.
        
        the samles will have the same mean and cov as the distribution bing sampled
        """
        
        return (
            numpy.hstack([
                numpy.matrix(numpy.random.randn(n,self.std.shape[0])),
                numpy.matrix(numpy.ones([n,1])),
            ])*diagstack([self.rotate,1])*self.affine
        )[:,:-1]

    def blend(*mvars):
        """
        This is awsome.
        
        optimally blend together any number of mvars.
        the process is *identical* to resistors in paralell, now that I have 
        a properly defined power functin (well I defined it to make this work)
        this function just calls a standard 'paralell' on the mvars and 
        operator overloading takes care of the rest
        
        when called as a method 'self' is part of *mvars 
        
        This blending function is not restricted to two inputs like the basic
        (wikipedia) version. Any number works.
        
        and it brings the symetry to the front. 
        
        >>>assert A.blend(B)== B.blend(A)== Mvar.blend(A,B)== 1/(1/A+1/B)

        the proof that this is identical to the wikipedia definition of blend 
        is a little too involved to write here. just try it (see the wiki 
        function below)
        """
        return paralell(mvars)
    
    def __and__(self,other):
        """
        this is just a shortcut to "blend"
        """
        return paralell((self,other))
    
    def __pow__(self,power):
        """
        I use this definition because 'blend' 
        becomes just a standard 'paralell'
        
        The main idea is that only the scale matrix gets powered.
        as the scale changes the mean changes with it.
                
        >>>assert (A**0).scale== eye
        >>>assert (A**0).rotate== A.rotate
        >>>assert (A**0).mean == A.mean*A.rotate.T*A.scale**-1*A.rotate
        
        >>>assert A**0== A**(-1)*A== A*A**(-1)== A/A        
        >>>assert A**K1*A**K2=A**(K1+K2)
        >>>assert A**K1/A**K2=A**(K1-K2)

        >>>assert A.std== A.scale*A.rotate
        >>>assert (A**K).std== (A.scale**K)*A.rotate
        >>>assert (A**K).std== A.std*A.rotate.T*A.scale**(K-1)*A.rotate
        >>>assert (A**K).mean== A.mean*A.rotate.T*A.scale**(K-1)*A.rotate
        >>>assert (A**K).affine== A.affine*A.rotate.T*A.scale**(K-1)*A.rotate

        because the scale matrix is a diagonal, powers on it are easy, 
        so this is not restricted to integer powers
        """
        rotate = self.rotate
        scale = numpy.diag(self.scale)
        undo=rotate.T*numpy.matrix(numpy.diag(scale**(-1)))
        new_std=numpy.matrix(numpy.diag(scale**(power)))*rotate
        
        return self*(undo*new_std)
    
    def __mul__(self,other):
                """
        Mvar*Mvar
            multiplying two Mvars together fits with the definition of power
            
            >>>assert A*A==A**2
            >>>assert (A*B).std == A.std*B.rotate.T*B.scale*B.rotate
            >>>assert (A*B).mean == A.mean*B.rotate.T*B.scale*B.rotate
            
            Note that the result does not depend on the mean of the second mvar(!)

        assert Mvar*constant == constant*Mvar
            Matrix multiplication and scalar multiplication behave differently 
            from eachother. 
            
            For this to be a properly defined vector space scalar multiplication 
            must fit with addition, and addition here is defined so it can be used 
            in the kalman noise addition step so: 
            
            assert lambda: (randn()+1)+(randn()+1) == lambda: sqrt(2)*randn() + 2
            
            assert (A+A).std == (2*A).std == sqrt(2)*A.std
            assert (A+A).mean == (2*A).mean == 2*A.mean
            
            assert (A*K).std == sqrt(K)*A.std
            assert (A*K).mean == K*A.mean
            
            assert (A*K).cov == A.cov*K
            
            be careful with negative constants because you will end up with 
            imaginary numbers in you std matrix, (and lime in your coconut) as 
            a direct result of:            
            
            assert B+(-A) == B+(-1)*A == B-A and (B-A)+A=B
            assert (A*K).std == sqrt(K)*A.std
            
            if you want to scale (like below) then use matrix multiplication
        
        Mvar*matrix

            matrix multiplication transforms the mean and ellipse of the 
            distribution
            
            assert lambda: 2*(randn()+1) == lambda 2*(randn())+2
            
            assert (A(*eye*K)).std == A.std*K
            assert (A(*eye*K)).mean == A.mean*K
            
            but any transform of the apropriate size works:
            
            assert (A*M).cov == M.T*A.cov*M
            assert (A*M).mean == A.mean*M

            matrix multiplication is implemented as follows
            
            assert A*M == Mvar(A.affine*diagstack([M,1])).refresh()
            
            the refresh() here is necessary to ensure that the rotation matrix
            stored in the object stays well behaved. 
        """
        other = self._mulconvert(other)
        return Mvar.multiplier[
            (type(self),type(other))
        ](self,other)
    
    def __rmul__(self,other):
        """
        be careful with right multiplying:
        Because power must fit with 
        multiplication
        
        assert A*A==A**2
        
        The most obvious way to treat right multiplication by a matrix is to 
        do exactly the same thing. So because of the definition of Mvar*Mvar
        (below)
        
        Mvar*Mvar
            multiplying two Mvars together fits with the definition of power
            
            assert prod(itertools.repeat(A,N)) == A**N
            assert A*B == A*(B.rotate.T*B.std) 
            
            the second Mvar is automatically converted to a matrix, and the 
            result is handled by matrix multiply
            
            again note that the result does not depend on the mean of the 
            second mvar(!)

        for consistancy when right multiplied, an Mvar is always converted to 
        the A.rotate.T*A.std matrix, and Matrix multiplication follows 
        automatically, and yields a matrix, not an Mvar.
        
        the one place this automatic conversion is not applied is when 
        right multiplying by a constant so: 
        
        martix*Mvar
            assert T*B == T*(B.rotate.T*B.scale*B.B.rotate)

        scalar multiplication however is not changed.
        
        assert Mvar*constant == constant*Mvar
        """
        other = self._mulconvert(other)
        return Mvar.multiplier[
            (type(other),type(self))
        ](other,self)
    
    def __imul__(self,other):
        """
        This is why I have things set up for left multply, it's 
        so that __imul__ works.
        """
        other = self._mulconvert(other)
        self.affine = Mvar.multiplier[
            (type(self),type(other))
        ](self,other).affine
    
    @staticmethod
    def _mulconvert(item): 
        return (
            Mvar if isinstance(item,Mvar) else 
            numpy.matix(item) if numpy.array(item).ndim else
            numpy.ndarray(item)
        )
        
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
            other*self.rotate.T*self.std
        ),
        lmul=lambda self,matrix: Mvar(
            self.affine*diagstack([mat, 1])
        ).refresh(),
    )
    
    def __div__(self,other):
        """
        see __mul__ and __pow__
        it would be immoral to overload power and multiply but not divide 
        assert A/B == A*(B**(-1))
        """
        return self*other**(-1)
        
    def __rdiv__(self,other):
        """
        see __rmul__ and __pow__
        assert A/B == A*(B**(-1))
        """
        return other*self**(-1)
        
    def __idiv__(self,other):
        self.affine=(self*other**(-1)).affine

    def __add__(self,other):
        """
        When using addition keep in mind that rand()+rand() is not like scaling 
        one random number by 2, it adds together two random numbers.

        The add here is like rand()+rand()
        
        Addition is defined this way so it can be used directly in the kalman 
        noise addition step
        
        so if you want simple scale use matrix multiplication like rand()*(2*eye)
        
        scalar multiplication however fits with addition:
        
        assert lambda: (randn()+1)+(randn()+1) == lambda: sqrt(2)*randn() + 2
        
        assert (A+A).std == (2*A).std == sqrt(2)*A.std
        assert (A+A).mean == (2*A).mean == 2*A.mean

        assert (A+B).mean== A.mean+B.mean
        assert (A+B).cov == A.cov+B.cov

        it also works with __neg__, __sub__, and scalar multiplication.
        
        assert B+(-A) == B+(-1)*A == B-A and (B-A)+A=B
        """
        try:
            return Mvar.from_mean_cov(
                mean= (self.mean+other.mean),
                cov = (self.cov + other.cov),
            )
        except AttributeError:
            return NotImplemented

    def __iadd__(self,other):
            self.affine = (self+other).affine

    def __sub__(self,other):
        """
        As with scalar multiplication and addition, watch out.
        
        here subtraction here is the inverse of addition 
         
            assert (A-B)+B == A
            assert (A-B).mean ==A.mean- B.mean
            assert (A-B).cov ==A.cov - B.cov
            
        if you want something that acts like rand()-rand() use:
            
            assert (A+B*(-1*eye)).mean == A.mean - B.mean
            assert (A+B*(-1*eye)).cov == A.cov + B.cov

        __sub__ also fits with __neg__, __add__, and scalar multiplication.
        
        assert B+(-A) == B+(-1)*A == B-A and (B-A)+A=B
        """
        try:
            return Mvar.from_mean_cov(
                mean= (self.mean-other.mean),
                cov = (self.cov - other.cov),
            )
        except AttributError:
            return NotImplemented
    
    def __isub__(self, other):
        self.affine = (self - other).affine

    def __neg__(self):
        """
        it would be silly to overload __sub__ without overloading __neg__
        
        assert B+(-A) == B+(-1)*A == B-A and (B-A)+A==B
        """
        return (-1)*self
    
    ################# Non-Math python internals
    def __str__(self):
        return ''.join('  Mvar(',self.affine.__str__(),')')
    
    def __repr__(self):
        return self.affine.__repr__().replace('matrix','  Mvar')

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
                    self.rotate,
                    complex,
                )).flat[0]
            ),
            #while transmitting any kwargs.
            **kwargs
        )

    
        
############### Helpers
def astype(self,newtype):
    duplicate=self
    duplicate.dtype=newtype
    return duplicate

def diagstack(arrays):
    """
    output is two dimensional

    type matches first input, if it is a numpy array or matrix, 
    otherwise this returns a numpy array
    """
    #latch the input iterable
    arrays = list(arrays)
    
    #get the type 
    atype = (
        type(arrays[0]) 
        if isinstance(arrays[0], numpy.ndarray) 
        else numpy.array
    )
    
    #convert each object in the list to a matrix
    arrays = list(numpy.matrix(array) for array in arrays)

    
    corners = numpy.vstack((
        #starting from zero
        (0,0),
        #get the index to where the upper right of each array will be
        numpy.cumsum(
            numpy.array([array.shape for array in arrays]),
            axis = 0
        )
    ))

    #make the block of zeros thatthe other arrays will be copied to
    result = numpy.zeros(corners[-1,:])
    
    #copy each arry into it's slot
    for (array,start,stop) in itertools.izip(arrays,corners[:-1],corners[1:]):
        result[start[0]:stop[0],start[1]:stop[1]] = array
    
    #set the array to the requested type
    return atype(result)

def autostack(lists):
    """
    simplify matrix stacking
    vertically stack the results of horizontally stacking each list

    hopefully in the future I will be able to overload this to automatically 
    call any callables in the list with sizes determined by their locations so 
    that
    
    autostack([
        [std,numpy.zeros]
        [mean,1]
    ])
    
    just works without having to explicitly declare the size of the zeros.
    
    """
    #interpret each item as a matrix
    lists= [[numpy.matrix(item) for item in row] for row in lists]
    
    #and do the stacking
    return numpy.vstack(
        numpy.hstack(
            numpy.matrix(item) 
            for item in row
        ) 
        for row in lists
    )


def paralell(items):
    """
    resistors in paralell, and thanks to 
    duck typing and operator overloading, this happens 
    to be exactly what we need for kalman style blending. 
    """
    inverted=[item**(-1) for item in items]
    return sum(inverted[1:],inverted[0])**(-1)


def matrix_constant_product(sequence):
    multiplier = reduce(
        function=mul,
        sequence=[
            item 
            for item in sequence 
            if item.ndim==0
        ],
        initial=numpy.array(1),
    )

    matrix= reduce(
        function= mul,
        sequence= [
            #while treating 0 dimensional arrays as consants and 
            #everything else as matrixes. a 1x1matrix is still 2d.
            numpy.matrix(item) 
            for item in sequence 
            if item.ndim!=0
        ],
        initial= numpy.array(1)
    )

    return (multiplier,matrix)
    

def wiki(P,M):
    """
    direct implementation of the wikipedia blending algorythm
    """
    yk=M.mean.T-P.mean.T
    Sk=P.cov+M.cov
    Kk=P.cov*(Sk**-1)
    
    return Mvar.from_mean_cov(
        (P.mean.T+Kk*yk).T,
        (numpy.eye(P.mean.size)-Kk)*P.cov
    )

def split(sequence,fkey=bool): 
    """
    return a defaultdict (where the default is an empty list), 
    where every value is a sub iterator produced from the sequence
    where items are sent to iterators based on the value of fkey(item).
    
    >>>isodd=split([1,2,3,4,5,6],lambda item:bool(item%2))
    >>>list(isodd[True])
    [1,3,5]
    >>>list(isodd[False])
    [2,4,6]
    
    this can also take the place of multiple passes with filter.
    """
    return collections.defaultdict(
        list,
        itertools.groupby(sequence,fkey),
    )
    
def product(*args):
    return reduce(
        function=operator.mul,
        sequence=args,
        initial=1
    )