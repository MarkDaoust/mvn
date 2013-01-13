import functools

import pylab
import matplotlib
import matplotlib.lines
import matplotlib.patches 

import mvn
import mvn.helpers as helpers
import numpy

class Plotter(object):
    def __init__(self,dist):
        self.dist = dist
    
    
    def plot(self, axis=None, **kwargs):
        """
        :param axis:
        :param ** kwargs:
        """
        
        defaults = self.defaultPlotParams.copy()
        defaults.update(kwargs)
        
        return self._plotter(axis, **defaults)
        

    @property
    def defaultPlotParams(self):
        """
        return a dictionary of default plot parameters
        """
        return {
            'edgecolor':'k'
        }

    @property
    def _plotter(self):
        """
        >>> if ndim >= 4:
        ...     assert A[:,:0]._plotter.im_func is Mvn.plotND
        ...     assert A[:,:1]._plotter.im_func is Mvn.plot1D
        ...     assert A[:,:2]._plotter.im_func is Mvn.plot2D
        ...     assert A[:,:3]._plotter.im_func is Mvn.plot3D
        ...     assert A[:,:4]._plotter.im_func is Mvn.plotND    
        """
        ndim = self.dist.ndim
        if ndim == 1:
            return self.plot1D
        
        if ndim == 2:
            return self.plot2D
        
        if ndim == 3:
            return self.plot3D

        return self.plotND
        

    def plot1D(self, 
        axis = None, 
        count = 1.0,
        fill = True, 
        nstd = 5,
        nsteps = 500, 
        orientation = 'horizontal',
        **kwargs
    ):
        """
        :param axis:
        :param count:
        :param fill:
        :param nstd:
        :param nsteps:
        :param orientation:
        :param ** kwargs:
        """                
        xlims = self.dist.bBox(nstd).squeeze()
        x = numpy.linspace(xlims[0], xlims[1], nsteps)
        y = count*self.dist.density(x[:, None])
        
        horizontal = ['horizontal', 'h', 'H']
        vertical   = ['vertical'  , 'v', 'V']
        
        if axis is None:
            axis = pylab.gca()        

        
        if fill:
            if orientation in horizontal:
                filler = axis.fill_betweenx
            elif orientation in vertical:
                filler = axis.fill_between
            else:
                raise ValueError(
                    'unknown orientation %s orientation should be in either (%r) or (%r)' % 
                    (orientation,horizontal, vertical)
                )
            plotter = functools.partial(filler, x, y, 0)
        else:
            plotter = functools.partial(axis.plot, x, y)
        
        return plotter(**kwargs)
                
    def plot2D(self, axis = None, nstd = 2, **kwargs):
        """
        :param axis:
        :param nstd:
        :param ** kwargs:
            
        plot a :py:meth:`mvn.Mvn.patch`, with axis autoscaling
        """
        if axis is None:
            axis = pylab.gca()
 
        bBox = self.dist.bBox(nstd).array()
        widths = numpy.diff(bBox, axis=0)
        pads = 0.05*widths*[[-1], [1]]
        corners = bBox+pads
        axis.update_datalim(corners)
        axis.autoscale_view()
                        
        artist = self.patch(**kwargs)
        
        if isinstance(artist, matplotlib.lines.Line2D):
            insert = axis.add_line
        else:
            insert = axis.add_patch
            
        return insert(artist)
        
    def plot3D(self, axis = None, nstd = 2.0, **kwargs):
        """
        :param axis:
        :param ** kwargs:
        """
        from mpl_toolkits.mplot3d import Axes3D

        if axis is None:
            axis = pylab.gca(projection = '3d')
            
        assert isinstance(axis, Axes3D)  
        
        
        u = numpy.linspace(0, 2 * numpy.pi, 100)
        v = numpy.linspace(0, numpy.pi, 100)

        x = numpy.outer(numpy.cos(u), numpy.sin(v))
        y = numpy.outer(numpy.sin(u), numpy.sin(v))
        z = numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))

        xyz = [x[..., None], y[..., None], z[..., None]]
    
        xyz = numpy.concatenate(xyz, -1)*nstd
        
        xyz = numpy.dot(
            xyz,
            self.dist.scaled.array()
        )

        xyz = xyz+self.mean.array()[None, :, :]
        
        axis.plot_wireframe(
            xyz[..., 0], xyz[..., 1], xyz[..., 2],  
            rstride=10, 
            cstride=10, 
            color = 'k',
            **kwargs
        )
            
        
    def plotND(self, axis = None, **kwargs):
        """
        :param axis:
        :param ** kwargs:
        """
        raise NotImplementedError()
       
        

    @classmethod
    def _kwargs2Marker(cls,**kwargs):
        facecolor = kwargs.pop('facecolor', None)
        if facecolor is not None:
            kwargs['markerfacecolor'] = facecolor
            
        edgecolor = kwargs.pop('edgecolor', None)
        if edgecolor is not None:
            kwargs['markeredgecolor'] = edgecolor
            kwargs['color'] = edgecolor
            
        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'

        return kwargs

    @classmethod
    def _convertAlpha(cls,color,alpha):

        if isinstance(color,str):
            colorConverter = matplotlib.colors.ColorConverter()
            color = colorConverter.to_rgb(color)

        color = list(color)

        if len(color) < 4:
            color.append(alpha)
        
        return color


    def patch(self, nstd=2, alpha='auto', slope=0.5, minalpha=0.3, **kwargs):
        """
        :param nstd:
        :param alpha:
        :param slope:
        :param minalpha:
        :param ** kwargs:
            
        get a matplotlib Ellipse patch representing the Mvn, all **kwargs are 
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
        shape = self.dist.shape

        assert shape[1] == 2,'this method can only produce patches for 2d data'
        
        if shape[0] < 2:
            kwargs = self._kwargs2Marker(**kwargs)   
            
            coords = self.dist.getX(nstd = nstd)
                    
            return matplotlib.lines.Line2D(
                coords[:, 0], 
                coords[:, 1], 
                **kwargs
            )
            

        if alpha == 'auto':
            alpha = numpy.max([
                minalpha,
                numpy.exp(-slope*mvn.sqrt(self.dist.det()))
            ])

            facecolor = kwargs.get('facecolor', None)
            if facecolor is None:
                kwargs['alpha'] = alpha
            else:
                kwargs['facecolor'] = self._convertAlpha(facecolor,alpha)

            edgecolor = kwargs.get('edgecolor', None)
            if edgecolor is not None:
                kwargs['edgecolor'] = self._convertAlpha(edgecolor,alpha)

        else:
            kwargs['alpha'] = alpha

        #unpack the width and height from the scale matrix 
        wh = nstd*mvn.sqrt(self.dist.var)
        wh[wh>1e5] = 1e5

        #convert from radius to diameters
        width,height = 2*wh

        #calculate angle
        angle = 180/numpy.pi*(
            numpy.angle(helpers.ascomplex(self.dist.vectors)[0])
        )        
        
        #return an Ellipse patch
        return matplotlib.patches.Ellipse(
            #with the Mvn's mean at the centre 
            xy=tuple(self.dist.mean.flatten()),
            #matching width and height
            width=width, height=height,
            #and rotation angle pulled from the vectors matrix
            angle=angle,
            #while transmitting any kwargs.
            **kwargs
        )

