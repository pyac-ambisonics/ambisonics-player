# Utility functions for Ambisonics processing.


import numpy as np


def ambix_channels_to_order(num_channels: int) -> int:

    if num_channels < 4:
       print(
        f"Channel count {num_channels} is too low for Ambisonics.\n"
        f"Minimum required: 4 channels (1st order)")
        
    order = int(round(np.sqrt(num_channels) - 1))


    if order < 1:
        order = 1
    
 
    expected = (order + 1) ** 2
    if num_channels != expected:
        print(f"{num_channels} channels → treating as order {order} (expects {expected} channels)")
        print("Empty channels will be auto-detected and ignored")
    
    return order




