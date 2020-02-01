import numpy as np


# class PyrmNet:
#     """
# 	inputs:
# 	inputs -> (4-D Tensor) with shape (None, 3, H, W) representing RGB images
# 
# 	parameters:
# 	n_gpu -> (non-negative integer) number of gpus available on machine
# 				can limit number of layers for GPU calcs
# 	gpu_only -> (boolean) whether layers should operate on CPU's if there
# 				is a need for GPUS
# 	center -> (boolean) whether the ArrayWeave includes center
# 	min_size -> (non-negative integer) the minimum dimension that ArrayWeave
# 				can operate on.
# 	n_layers -> (non-negative integer) the numbers of layers in a pyrm-net:
# 				n_layers = min(n_layers,
# 				 			fn(gpu_only,n_gpu) #Can we use gpus? if so how big?
# 				 			fn(max_pool_loc,min_size)) #how small can we make the image?
# 	n_filters -> (non-negative integer) the number of filters on the first layer
# 	r_filters -> (non-negative integer) the ratio at which the number of filters
# 				changes as layers are advanced
# 	max_pool_loc -> (non-negative integer) the number of layers passed since
# 				the first layer for the next MaxPool operation
# 	end_max_pool -> (boolean) whether a max_pool operat
# 	"""
#
#     def __init__(self,
#                  n_layers,
#                  n_filters_start,
#                  n_gpus,
#                  image_size,
#                  r_filter=2,
#                  r_combine=2,
#                  max_pool_loc=2,
#                  end_max_pool=True,
#                  min_dim=8,
#                  center=False,
#                  gpu_only=False):
#
#         self.n_layers = min(n_layers, max(gpu_only * int(np.log2(n_gpus)), (1 - gpu_only) * float('inf')),
#                             min_dim * 2 ** ((n_layers / max_pool_loc) + end_max_pool))
#
#         self.n_filters_start = n_filters_start
#         self.n_gpus = n_gpu
#         self.r_filter = r_filter
#         self.r_combine = r_combine
#         self.max_pool_loc = max_pool_loc
#         self.end_max_pool = end_max_pool
#         self.min_dim = min_dim
#         self.center = center
#
#     	self.gpu_only = gpu_only
