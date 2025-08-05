from materialEncoder import MaterialEncoder
from networks import VariationalAutoencoder 
from smallestEllipse import plot_ellipse, welzl
import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append('../PyTO/src') #assuming the PyTO is in the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
from mmaWrapper import runMMA
from topopt_common import *
from topopt_material_model import *
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import mat_lib

import bound_cond
import hex_mesher
