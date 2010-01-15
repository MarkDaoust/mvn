import itertools

import numpy

class Mvar(object):
    """
    Multivariate normal distributions packaged to act like a vector.
        
    This is done with kalman filtering in mind, but is good for anything where 
        you need to track linked uncertianties across multiple variables.
    
    basic math (+,-,*,**) has been overloaded to work normally.
        
    The data is stored as an affine transformation, that is one large matrix 
    containing the mean and standard-deviation matrixes.
    
        affine = autostack([
            [ std,0],
            [mean,1],
        ])
        
        The "standard deviation" matrix is a scaled eigenvector matrix
        where each -row- is a scaled vector?
        it will make compression easier (in the future) but certian other 
        things harder    
        
        you can think of *everything* as being embeded on a sheet in a space 
        with 1 extra dimension whih the sheet is located 1 unit "up" along the 
        extra dimension.
        
        with this setup simple math operations (+,-,*,**) are can be done 
        almost directly on the affine transforms with little extra work.
    
    the from* functions all create new instances from varous 
        common constructs.
        
    the get* functions all grab useful things out of the structure, 
        all have equivalent parameters linked to them
    """
    
    ############## Creation
    def __init__(self,affine):
        """
        create an instance directly from an affine transform.
        the transform should be of the format:
        
        affine = autostack([
            [ std,0],
            [mean,1],
        ])
            
        the affine transform is the only real property in these things 
        everything else is calculated on the fly
        
        standard deviation is not a common term when dealing with these things,
        let me explain:
        
        #scale is a diagonal matrix of eigenvalues
        scale=diag(sqrt(eigenvalues(cov)))
        #rotate is a matrix of eigenvectors (colums)
        rotate=eigenvectors(cov)
        
        std==scale*rotate
        std**(-1)==rotate.T*scale**(-1)
        
        rotate.T == rotate**(-1)
        
        also note that: 
            std.T*std == rotate.T*scale*scale*rotate == cov
            std*std.T == scale*rotate*rotate.T*scale ==scale**2
        """
        self.affine=affine
    
    ############## alternate creation methods
    #maybe eventually I'll link these to the properties 
    #to make them all writeable
    
    @staticmethod
    def from_mean_std(mean=None,std=None):
        """
        if either one is missing it gets filled with zeros,
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
        
        remember numpy's default is to divide by (n-1) not (n) set bias = 1
        to normalize by 1
        """
        #convert the data to a matrix 
        data=numpy.matrix(data)
        
        #create the mvar from the mean and covariance of the data
        return Mvar.from_mean_cov(numpy.mean(data), numpy.cov(data, bias=bias))
    
    @staticmethod
    def from_roto_scale(rotation,scale,mean=None):
        """
        Rotation can be either a matrix or, if the mean and scale are 
        two dimensional, it can be the rotation angle and the rotation matrix 
        will be created automatically.
        
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
        mean = zeros((1,scale.shape[0])) if mean is None else mean
        
        #create the Mvar
        return Mvar(autostack([
            [ scale*rotation, numpy.zeros([scale.shape[0],1])],
            [mean, 1],
        ]))

    @staticmethod
    def stack(*mvars):
        """
        yes it works but be careful. Don't use this for reconnecting 
        something you've just calculated back to the data it was calculated 
        from. If you're trying to do that use a better matrix multiply, 
        so you don't loose the cross corelations
        """
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
        A.rotate*A.rotate.T == eye
        """
        return autostack([
            vector/numpy.sqrt(vector*vector.T) 
            for vector in self.std
        ])
        

    def get_scale(self):
        """
        get the scale matrix used in the object
        aka a diagonal matrix of eigenvalues (
        
        assert A.std == A.scale*A.rotate
        assert A.scale**2 == A.std.T*A.std
        assert A.cov == A.rotate*A.scale**2*A.rotate
        """
        return numpy.matrix(numpy.diag(numpy.sqrt(numpy.diag(
            #when multiplied like this the rotation matrixes cancle out and 
            #all you're left with is the square of the scales on the diagonal
            #std=scale*rotate
            #std*std.T=scale*rotate*rotate.T*scale=scale*scale
            self.std*self.std.T
        ))))


    def get_cov(self):
        """
        get the covariance matrix used by the object
        assert A.cov == A.std.T*A.std 
        assert A.cov == A.rotate.T*A.scale.T*A.scale*A.rotate 
        assert A.cov == A.rotate.T*A.scale**2*A.rotate
        """
        return self.std.T*self.std
    
    def get_mean(self):
        """
        get the mean of the distribution (row vector)
        """
        return self.affine[-1,:-1]
    
    def get_std(self):
        """
        get the standard deviation of the distribution,
        aka matrix of scaled eigenvectors (as rows)
        std = scale*rotate 
        """
        return self.affine[:-1,:-1]
    
    ############ Properties
    #maybe later I'mm add in some of those from functions
    rotate= property(fget=get_rotate)
    scale = property(fget=get_scale)
    cov   = property(fget=get_cov)
    mean  = property(fget=get_mean)
    std   = property(fget=get_std)

    #I'm not sure if it's reasonable to keep this....
    @property
    def T(self):
        """
        return an Mvar with the affine transform transposed
    
        I'm still not sure what this implies...
        
        it seems like the right way to transpose the state vector...
        but how come I can do multiplication with matracies that would be 
        illegal in nomal space
        
        oh and if it's transposed get_mean can't find the mean
        
        but it does make __rmul__ easier
        """
        return Mvar(self.affine.T)
    
    ############ Math
    def blend(*mvars):
        """
        blend any number of mvars.
        the process is identical to resistors in paralell.
        this function just calls 'paralell' on the mvars and operator 
        overloading takes care of the rest
        
        when called as a method 'self' is part of *mvars 
        
        This blending function is not restricted to two inputs like the basic
        (wikipedia) version. Any number works. 
        """
        return paralell(mvars)
        
    def __pow__(self,power):
        """
        I use this definition because 'blend' 
        becomes just a standard 'paralell'
        
        The main idea is that only the scale matrix gets powered.
        as the scale changes the mean changes with it.
        
        assert (A**0).scale==eye
        assert (A**0).rotate==A.rotate
        
        assert A.std==A.scale*A.rotate
        assert (A**K).std==(A.scale**K)*A.rotate
        
        assert A**0.scale == eye(A.std.shape[2])
        assert A**0.std == A.rotate
        
        because the scale matrix is a diagonal, powers on it are easy, 
        so this is not restricted to integer powers
        """
        rotate = self.rotate
        scale = numpy.diag(self.scale)
        undo=rotate.T*numpy.matrix(numpy.diag(scale**(-1)))
        new_std=numpy.matrix(numpy.diag(scale**(power)))*rotate
        
        return self*(undo*new_std)
        
    def __mul__(self, other):
        """
        multiplying two Mvars together fits with the definition of power
        
        assert A*A==A**2
        assert (A*B).std == A.std*B.rotate.T*B.scale*B.rotate
        assert (A*B).mean == A.mean*B.rotate.T*B.scale*B.rotate

        Note that the result does not depend on the mean of the second mvar(!)
        
        Matrix multiplication and scalar multiplication behave differently 
        from eachother. For this to be a properly defined vector space scalar 
        multiplication must fit with addition. 
        and addition here is defined so it can be used in the kalman 
        noise addition step so: 
        
        lambda: (randn()+1)+(randn()+1) == lambda: sqrt(2)*randn() + 2
        
        assert (A+A).std == (2*A).std == sqrt(2)*A.std
        assert (A+A).mean == (2*A).mean == 2*A.mean
        
        assert (A*K).std == sqrt(K)*A.std
        assert (A*K).mean == K*A.mean
        
        if you want to scale (like below) then use matrix multiplication
        matrix multiplication transforms the mean and ellipse of the 
        distribution
        
        lambda: 2*(randn()+1) == lambda 2*(randn())+2
        
        assert (A(*eye*K)).std == A.std*K
        assert (A(*eye*K)).mean == A.mean*K
        
        but any transform of the apropriate size works:
        
        assert (A*T).std == A.std*T
        assert (A*T).mean == A.mean*T
        """
        other = (
            #get the round trip transform
            other.rotate.T*other.std
            #if we're multiplying by another mvar
            if isinstance(other,Mvar)
            #if we're  multiplying by anything else convert it to an array 
            else numpy.array(other)
        )
        
        #if the array is zero dimensional 
        #(a 1x1 numpy.matrix is 2 dimensional)
        if other.shape == ():
            #do scalar multiplication
            return Mvar.from_mean_std(
                mean= self.mean*other,
                std = self.std*numpy.sqrt(other),
            )
        else:
            #convert the other to a matrix, and append a 1 to the bottom 
            #right corner 
            other=diagstack([numpy.matrix(other), 1])
            #then do the multiplication
            return Mvar(self.affine*other)
    
    def __rmul__(self,other):
            other = numpy.matrix(other)
            return Mvar.((self.rotate.T*self.std)*other.T).T
        

    def __imul__(self,other):
        #if the other is iterable  
        if hasattr(other,'__iter__'):
            other=diagstack([matrix(other),1])
            self.affine = self.affine*other
        else:
            #do scalar multiplication
            self.affine = Mvar.from_mean_std(
                mean= self.mean*other,
                std = self.std*numpy.sqrt(other),
            ).affine

    def __add__(self,other):
        """
        When using addition keep in mind that rand()+rand() is not like scaling 
        one random number by 2, it adds together two random numbers.
        
        the add here is like rand()+rand()
        
        If you want to scale it use matrix multiplication rand()*eye*2
        
        scalar multiplication however fits with addition:
            rand()+rand() == 2*rand()
            
        C=A+B
        assert C.mean== A.mean+B.mean
        assert C.cov == A.cov+B.cov
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
        
        subtraction here is the inverse of addition 
        (if A and B are both Mvars) 
            
            C1=A-B
            assert C1+B==A
            assert C1.mean==A.mean- B.mean
            assert C1.cov ==A.cov - B.cov
            
        if you want something that acts like rand()-rand() use:
            
            C2 = A+B*(-eye)            
            assert C2.mean== A.mean - B.mean
            assert C2.cov == A.cov + B.cov
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
    
    ################# Non-Math python internals
    def __str__(self):
        return ''.join('Mvar(',self.affine.__str__(),')')
    
    def __repr__(self):
        return self.affine.__repr__().replace('matrix','Mvar')
        
############### Helpers
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
    result = numpy.zeros(shapes[-1,:])
    
    #copy each arry into it's slot
    for (array,start,stop) in itertools.izip(arrays,shapes[:-1],shapes[1:]):
        result[start[0]:stop[0],start[1]:stop[1]] = array
    
    #set the array to the requested type
    return atype(result)

def autostack(lists):
    """
    simplify matrix stacking
    vertically stack the results of horizontally stacking each list
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
