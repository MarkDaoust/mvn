#! /usr/bin/env python

import functools

try:
    from decorator import decorator
except ImportError:
    class decorator(object):
        def __init__(self,decorator):
            self.decorator=decorator
        def __call__(self,decorated):
            return functools.partial(decorator,decorated)
