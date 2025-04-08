# Einops

Parse the numpy arrays and implement the Rearrange function from the Einops library

- `parser.py` - Contains the expression parser to map the expression to the shape dictionary and handle ellipsis
- `funcs.py` - Contains the main `rearrange` function to reshape, transpose and split, merge or repeat axes.
- `unit_tests.py` - Unit tests covering different usecases and edge cases
- `Einops.ipynb` - Notebook for same implementation

### Requirements
- Python >= 3.8
- [numpy](https://pypi.org/project/numpy/)

### How to use
- To use the `rearrange` function:
```
import numpy as np
from funcs import rearrange
x = np.arange(120).reshape(2,4,5,3)     #define input tensor
y = rearrange(x, 'b c h w -> b h w c')      #rearrange as per the pattern defined
```

- To run unit tests:
```
python3 unit_tests.py
```

### Design decisions and approach
- Tokenize the expression in parser to reduce the operations
- Map the extracted tokens to get the composition i.e. elements present in the expression
- Parser uses identifiers(set) for dimension tracking
- Handle the input and output shapes as per expression to pass on for numpy reshape function
- Map the tensor sizes to the expression dimesnsion
- Handle ellipsis passing '_ellipsis_' in the composition
- Three-step process (ungroup → reorder → group) with respect to intermediate_shape, permutation, and final_shape
- Handles error cases and complex patterns
