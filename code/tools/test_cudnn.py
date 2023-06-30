import ctypes
import os

# Load the libcudnn library
try:
    cudnn = ctypes.CDLL("libcudnn.so")
    print("libcudnn library loaded successfully.")

    # Get the loading path of the libcudnn library
    lib_name = "libcudnn.so"
    lib_path = None
    for path in os.environ["LD_LIBRARY_PATH"].split(":"):
        full_path = os.path.join(path, lib_name)
        if os.path.exists(full_path):
            lib_path = full_path
            break
    
    if lib_path:
        print("Loading path:", lib_path)
    else:
        print("Loading path not found.")
        
except OSError:
    print("Failed to load libcudnn library.")
