import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupSort(nn.Module):
    def __init__(self, num_units, axis=-1):
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        group_sorted = group_sort(x, self.num_units, self.axis)
        assert check_group_sorted(group_sorted, self.num_units, axis=self.axis) == 1, "GroupSort failed. "
        return group_sorted

    def extra_repr(self):
        return 'num_groups: {}'.format(self.num_units)


def process_group_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size

def group_sort(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*list(x.shape))
    return sorted_x

def check_group_sorted(x, num_units, axis=-1):
    size = process_group_size(x, num_units, axis)

    x_np = x.cpu().data.numpy()
    x_np = x_np.reshape(*size)
    axis = axis if axis == -1 else axis + 1
    x_np_diff = np.diff(x_np, axis=axis)

    # Return 1 iff all elements are increasing.
    if np.sum(x_np_diff < 0) > 0:
        return 0
    else:
        return 1
    
    
class TextClassifier(nn.ModuleList):

	def __init__(self, params, size_tile):
		super(TextClassifier, self).__init__()

		# Parameters regarding text preprocessing
		self.seq_len = params.seq_len
		self.embedding_size = int(size_tile)
		#self.num_words = params.num_words
		
		# Dropout definition
		self.dropout = nn.Dropout(0.50)
		
		# CNN parameters definition
		# Kernel sizes
		self.kernel_1 = 2
		self.kernel_2 = 3 #not used in the current solution
		self.kernel_3 = 10 #not used in the current solution
		self.kernel_4 = 20
		
		# Output size for each convolution
		self.out_size = params.out_size
		# Number of strides for each convolution
		self.stride = params.stride
		
		# Embedding layer definition
		#self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)
		
		# Convolution layers definition
		self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
		self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
		self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
		self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)
		
		# Max pooling layers definition
		self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
		self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
		self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
		self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)
		
		# Fully connected layer definition
		print(self.in_features_fc())
		self.fc = nn.Linear(self.in_features_fc(), 1)
		self.fc2 = nn.Linear(32, 1)

		
	def in_features_fc(self):
		'''Calculates the number of output features after Convolution + Max pooling
			
		Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
		Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
		
		source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
		'''
		# Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
		out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
		out_conv_1 = math.floor(out_conv_1)
		out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
		out_pool_1 = math.floor(out_pool_1)
		
		# Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
		out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
		out_conv_2 = math.floor(out_conv_2)
		out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
		out_pool_2 = math.floor(out_pool_2)
		
		# Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
		out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
		out_conv_3 = math.floor(out_conv_3)
		out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
		out_pool_3 = math.floor(out_pool_3)
		
		# Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
		out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
		out_conv_4 = math.floor(out_conv_4)
		out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
		out_pool_4 = math.floor(out_pool_4)
		
		# Returns "flattened" vector (input for fully connected layer)
		print(out_pool_1, out_pool_2, out_pool_3, out_pool_4)        
		#return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size
		return (out_pool_1 + out_pool_4) * self.out_size
		
		
	def forward(self, x):
		# Sequence of tokes is filterd through an embedding layer
		x = x.float()
		x1 = self.conv_1(x)
		x1 = self.pool_1(x1)
		
		# Convolution layer 2 is applied
		x2 = self.conv_2(x)
		x2 = self.pool_2(x2)

		# Convolution layer 3 is applied
		x3 = self.conv_3(x)
		x3 = self.pool_3(x3)
		
		# Convolution layer 4 is applied
		x4 = self.conv_4(x)
		x4 = self.pool_4(x4)
		
		# The output of each convolutional layer is concatenated into a unique vector
		union = torch.cat((x1, x4), 2)
		union = union.reshape(union.size(0), -1)

		# The "flattened" vector is passed through a fully connected layer
		out = GroupSort(num_units=1)(union)
		out = nn.Dropout(0.95)(out)
		out = self.fc(out)
		out = torch.sigmoid(out)
		return out.squeeze()