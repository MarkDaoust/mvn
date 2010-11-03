import numpy
from mvar import Mvar
from matrix import Matrix
import cPickle
locals().update(
    cPickle.load(open("testObjects.pkl","r"))
)