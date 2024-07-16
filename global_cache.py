import os
import numpy
from easydict import EasyDict

C = EasyDict()
global_cache = C

C.RWPM = 1
C.CALIBRATION = 0
C.alpha = 0.99
C.TT = 5
C.temperture = 0.01
C.n_patrion = 2