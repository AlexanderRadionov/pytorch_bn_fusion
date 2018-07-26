from .bn_fusion import fuse_bn_sequential, fuse_bn_recursively
from .utils import convert_resnet_family, convert_resnet_family_recursively

__version__ = '1.0.0'
__all__ = [fuse_bn_sequential, fuse_bn_recursively, convert_resnet_family, convert_resnet_family_recursively]
