from .logger import Logger, create_file_logger
from .misc import get_config

__all__ = ['Logger', 'get_config', 'create_file_logger']


import sys


_LIBS = ['./external/packnet_sfm', './external/dgp', './external/monodepth2']    

def setup_env():       
    if not _LIBS[0] in sys.path:        
        for lib in _LIBS:
            sys.path.append(lib)

setup_env()