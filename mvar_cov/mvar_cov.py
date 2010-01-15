import itertools

import numpy

global __debug__

#consider storing this as eigenvectors*eigenvalues,
#it would make multipy's, transposes and most transforms (scale/rotate) easier.
#at a cost to add and subtract I think it's worth it. 

class Mvar(object):
    """
    Multivariate normal distributions packaged to act like a vector.
        
    This is done with kalman filtering in mind, but is good for anything where 
        you need to track linked uncertianties across multiple variables.
    
    basic math has been overloaded to work normally.
    but keep in mind that 2*rand() != rand()+rand()    
        if you want it to act like 2*rand() do 2*eye*rand() (scaleing the whole thing)
        if you want rand()+rand() do 2*Mvar or Mvar+Mvar (add them together)
    
    The data is stored as an affine transformation:
        one large matrix containing the mean and standard-deviation matrixes.
        you can think of everything as being embeded on a sheet in a space 
        with 1 extra dimension the sheet is located 1 unit "up" along the 
        extra dimension
        
        with this setup simple math operations (+,-,*,**) are can be done directly 
        on the affine transforms with almost no other work.
    
    the from* functions all create new instances from varous 
        common types of data.
    
    the get* functions all manage the eigenvectors/eigenvalues, AKA scale and 
        rotation matrixes  
    """

    ############## Creation
    def __init__(self,affine):
        """
        create an instance directly from an affine transform.
        the transform should be of the format:
        
            [[ cov],[0]]
            [[mean],[1]]
        
        it's too bad numpy matrix stacking can't handle that directly
        
        the affine transform is the only real property in these things 
        everything else is calculated on the fly
        """
        self.affine=affine
    
    ############## alternate creation methods
    
    @staticmethod
    def from_mean_cov(mean = None,cov = None):
        if mean is None:
            cov = zeros(mean.size,mean.size)
        elif cov is None:
            mean = zeros(cov.shape)
        
        #localize numpy
        N = numpy
        
        #convert inputs to matrixes 
        cov = N.matrix(cov)
        mean = N.matrix(mean)
        
        T=False
        if mean.shape[0]>mean.shape[1]:
            T=True
            mean=mean.T
        
        #create the affine transform, and from it the Mvar
        result = Mvar(
            N.vstack([
                N.hstack([ 
                    cov,
                    N.matrix(N.zeros([len(cov),1])),
                ]),
                N.hstack([ 
                    mean,
                    N.matrix(1),
                ]),
            ])
        )
        
        return result.T if T else result 
    
    @staticmethod
    def from_data(data):
        """
        create an Mvar with the same mean and covariance as the supplied data
        with each row being a sample and each column being a dimenson
        
        remember numpy's default is to divide by (n-1) not (n)
        """
        #localize numpy
        N=numpy
        
        #convert the data to a matrix 
        data=N.matrix(data)
        
        #create the mvar from the mean and covariance of the data
        return Mvar.from_mean_cov(N.mean(data),N.cov(data))
    
    @staticmethod
    def from_roto_scale(rotation,scale,centre):
        """
        rotaton and scale, scale line standard deviation not variance. 
        """
        
        #Localize numpy 
        N=numpy
        
        #convert everything to matrixes
        rotation,scale,centre=(
            matrix(data) 
            for data in (rotation,scale,centre)
        )
        
        #if the scale matrix is not square
        if scale.shape[0] != scale.shape[1]:
            #it should be a vector the same shape as the centre
            assert scale.shape==centre.shape
            #convert it to a diagonal matrix
            scale=matrix(N.diag(N.array(scale)))
        
        #if the rotatin matrix is a scalar, and we're working in 2 dimensions
        if rotation.size==1 and scale.shape==(2,2):
            # then rotation is the rotation angle, 
            #create the apropriate rotation matrix
            rotation=matrix([
                [ N.cos(rotation),N.sin(rotation)],
                [-N.sin(rotation),N.cos(rotation)],
            ])
        
        #create the covariance matrix and, from that, the Mvar 
        return Mvar.from_mean_cov(centre,rotation*scale*scale*rotation.T)

    @staticmethod
    def from_guess(thing):
        
        thing=numpy.matrix(thing)

        if other.shape[0]!=other.shape[1] or max(other.shape)==1:
            return Mvar.from_mean_cov(mean=other)
        else:
            return Mvar.from_mean_cov(cov=other) 
    
    @staticmethod
    def stack(mvars):
        """
        yes it works but be careful. Don't use this for reconnecting 
        something you've just calculated back to the data it wa salculated from. If you're
        trying to do that use a better matrix multiply, so you don't loose 
        the cross corelations
        """
        return Mvar.from_mean_cov(
            #stack the vectors horizontally
            numpy.hstack([mvar.mean for mvar in mvars]),
            #stack the covariances diagonally
            diagstack([mvar.cov for mvar in mvars]),
        )

    ############ Eigen-stuf
    def get_rotate(self):
        #the rotation matrix is the matrix of eigenvectors
        return numpy.linalg.eigh(self.cov)[1]

    def get_scale(self):
        #the scale matrix is just the eigenvalues along the diagonal
        #the eigenvalues are for the covariance matrix, 
        #which is the square of the standard deviations..
        return numpy.matrix(numpy.diag(
            numpy.linalg.eigvalsh(self.cov)**0.5
        ))

    def get_scale_rotate(self):
        #it's more efficient to calculate these together
        values,vectors = numpy.linalg.eigh(self.cov)
        values = numpy.matrix(numpy.diag(values**0.5))
        return (values,vectors)
        
    def get_std(self):
        #the scale in the middle is square-rooted so this acts 
        #like a standard deviation instead of a variance
        (scale,rotate) = self.get_scale_rotate()
        return rotate*scale*rotate.T
    
    def get_mean(self):
        return self.affine[-1,:-1]
    
    def get_cov(self):
        return self.affine[:-1,:-1]
    
    
    ############ Properties
    mean= property(fget=Mvar.get_mean)
    cov = property(fget=Mvar.get_cov)
    
    @property
    def T(self):
        return Mvar(self.affine.T)
    
        
    ############ Iteration
    def __iter__(self):
        return __iter__(self.affine)
    
    ############ Math
    def blend(*mvars):
        """
        'self' is part of *mvars 
        
        This blending function is not restricted to two inputs like the basic
        (wikipedia) version. 
        
        am I extending it correctly?
        """
        return paralell(mvars)
        
    def __pow__(self,power):
        """
        I use this definition because 'blend' 
        becomes just a standard 'paralell'
        """
        return self*self.cov**(power)

    def __add__(self,other):
        """
        when using addition keep in mind that rand()+rand() != 2*rand()
        if you want to scale it use matrix multiplication
        """
        if isinstance(other,Mvar):
            return Mvar(self.affine+other.affine)
        else:
            return self+Mvar.from_guess(other)
    
    def __radd__(self,other):
        return self+other
    
    def __iadd__(self,other):
        if isinstance(other,Mvar):
            self.affine+=other.affine
        else:
            self+=Mvar.from_guess(other).affine
    
    def __sub__(self,other):
        if isinstance(other,Mvar):
            return Mvar(self.affine-other.affine)
        else:
            return self-Mvar.from_guess(other)
        
    def __rsub__(self,other):
        return Mvar.from_guess(other)-self
    
    def __isub__(self,other):
        if isinstance(other,Mvar):
            self.affine-=other.affine
        else:
            self-=Mvar.from_guess(other)
    
    def __mul__(self,other):
        if isinstance(other,Mvar):
            return self*other.cov
        else:
            #if the other is iterable  
            if hasattr(other,'__iter__'):
                #attempt to convert it to a matrix, then appen a 1 to 
                #the bottom right corner 
                other=diagstack([matrix(other),1])
                #and do the multiplication
                return Mvar(other.T*self.affine*other)
            else:
                #do scalar multiplication
                return Mvar(self.affine*other)
    
    def __rmul__(self,other):
        return self*other.T
    
    def __imul__(self,other):
        if isinstance(other,Mvar):
            self*=other.cov
        else:    
            #if the other is iterable  
            if hasattr(other,'__iter__'):
                other=diagstack([matrix(other),1])
                self.affine = other.T*self.affine*other
            else:
                #do scalar multiplication
                self.affine *= other

    def __neg__(self):
        return (-1)*self
    
############### Helpers
def diagstack(arrays):
    """
    output is two dimensional

    type matches first input, if it is a numpy array or matrix, 
    otherwise this returns a numpy array
    """
    arrays = list(arrays)
    
    atype = (
        type(arrays[0]) 
        if isinstance(arrays[0], (numpy.array, numpy.matrix)) 
        else numpy.array
    ) 
    
    arrays = list(numpy.matrix(array) for array in arrays)

    shapes = numpy.vstack((
        (0,0),
        numpy.cumsum(
            numpy.array([array.shape for array in arrays]),
            axis = 0
        )
    ))

    result = numpy.zeros(shapes[-1,:])
    
    for (array,start,stop) in itertools.izip(arrays,shapes[:-1],shapes[1:]):
        result[start[0]:stop[0],start[1]:stop[1]] = array
    
    return atype(result)


def paralell(items):
    #resistors in paralell
    return sum(item**(-1) for item in items)**(-1)
