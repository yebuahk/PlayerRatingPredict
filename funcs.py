import numpy as np

def recode_position (position):
    if position == 'ST':
        return 1
    elif position == 'CB':
        return 0
    else:
        return np.nan
