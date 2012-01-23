
import sys
import numpy
import matplotlib

matplotlib.use(sys.argv[1])

import pylab

data = numpy.random.randn(10000)

pylab.hist(data,200)

try:
    pass
#    pylab.savefig('test.png',format = 'png')
except:
    pass

try:
    pass
#    pylab.savefig('test.svg',format = 'svg')
except:
    pass

pylab.show()

