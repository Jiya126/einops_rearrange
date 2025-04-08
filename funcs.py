import numpy as np
from typing import List, Dict, Tuple

from parser import ParsedExpression, EinopsError

def _get_shape_dict(tensor: np.ndarray, source_parsed: ParsedExpression, named_sizes: Dict[str, int]) -> Dict[str, int]:
    """
    Create a dictionary mapping dimension names to their sizes.
    Arguments:
        tensor (np.ndarray): The input tensor
        source_parsed (ParsedExpression): The parsed expression
        named_sizes (Dict[str, int]): Additional named dimensions and their sizes
    Returns:
        shape_dict (Dict[str, int]): Dictionary mapping dimension names to their sizes
    """
    shape_dict = named_sizes.copy()
    current_dim = 0
    
    for item in source_parsed.composition:
        if isinstance(item, list):
            total_size = tensor.shape[current_dim]
            
            if len(item) == 1:
                axis_name = item[0]
                if axis_name not in shape_dict:
                    shape_dict[axis_name] = total_size
                elif shape_dict[axis_name] != total_size:
                    raise EinopsError(f"Inconsistent size for dimension {axis_name}: got {total_size}, expected {shape_dict[axis_name]}")
                current_dim += 1
                continue
            
            unknown_dims = tuple(dim for dim in item if dim not in shape_dict)
            
            if not unknown_dims:
                product = np.prod(tuple(shape_dict[dim] for dim in item))
                if product != total_size:
                    raise EinopsError(f"Shape mismatch: {item} product {product} != {total_size}")
            elif len(unknown_dims) == 1:
                known_dims = tuple(shape_dict[dim] for dim in item if dim in shape_dict)
                known_product = np.prod(known_dims)
                if total_size % known_product != 0:
                    raise EinopsError(f"Cannot divide dimension size {total_size} by {known_product}")
                shape_dict[unknown_dims[0]] = total_size // known_product
            else:
                raise EinopsError(f"Cannot infer sizes for multiple unknown dimensions in {item}")
            current_dim += 1
            
        elif item == "_ellipsis_":
            remaining_dims = sum(1 for x in source_parsed.composition[source_parsed.composition.index(item)+1:] 
                               if isinstance(x, list))
            ellipsis_dims = len(tensor.shape) - current_dim - remaining_dims
            if ellipsis_dims < 0:
                raise EinopsError("Pattern has more dimensions than tensor")
            current_dim += ellipsis_dims
    
    return shape_dict

def _compute_output_shape(tensor: np.ndarray, target_parsed: ParsedExpression, 
                         source_parsed: ParsedExpression, shape_dict: Dict[str, int]) -> Tuple[List[int], List[int], List[int]]:
    """
    Compute shapes and permutations for the output tensor.
    Arguments:
        tensor (np.ndarray): The input tensor
        target_parsed (ParsedExpression): The parsed target expression
        source_parsed (ParsedExpression): The parsed source expression
        shape_dict (Dict[str, int]): Dictionary mapping dimension names to their sizes
    Returns:
        final_shape (List[int]): Final target shape after regrouping dimensions
        permutation (List[int]): List of indices showing how dimensions should be reordered
        intermediate_shape (List[int]): Shape after ungrouping dimensions but before permutation
    """
    intermediate_shape = []
    final_shape = []
    source_positions = {}
    permutation = []
    current_dim = 0
    
    # Handle source pattern
    for item in source_parsed.composition:
        if isinstance(item, list):
            if len(item) == 1:
                intermediate_shape.append(shape_dict[item[0]])
                source_positions[item[0]] = current_dim
                current_dim += 1
            else:
                for axis in item:
                    intermediate_shape.append(shape_dict[axis])
                    source_positions[axis] = current_dim
                    current_dim += 1
        elif item == "_ellipsis_":
            ellipsis_start = current_dim
            ellipsis_end = len(tensor.shape) - sum(1 for x in source_parsed.composition[source_parsed.composition.index(item)+1:] if isinstance(x, list))
            ellipsis_dims = list(range(ellipsis_start, ellipsis_end))
            intermediate_shape.extend(tensor.shape[ellipsis_start:ellipsis_end])
            current_dim = ellipsis_end
    
    # Build output shape and permutation
    for item in target_parsed.composition:
        if isinstance(item, list):
            if len(item) == 1:
                if item[0] not in shape_dict:
                    raise EinopsError(f"Unknown dimension: {item[0]}")
                final_shape.append(shape_dict[item[0]])
                permutation.append(source_positions[item[0]])
            else:
                size = np.prod(tuple(shape_dict[axis] for axis in item))
                final_shape.append(size)
                for axis in item:
                    permutation.append(source_positions[axis])
        elif item == "_ellipsis_":
            final_shape.extend(tensor.shape[d] for d in ellipsis_dims)
            permutation.extend(ellipsis_dims)
    
    # Validate final shape
    if np.prod(tuple(intermediate_shape)) != np.prod(tuple(final_shape)):
        raise EinopsError(f"Cannot reshape array of size {np.prod(tuple(intermediate_shape))} into shape {tuple(final_shape)}")
    
    return intermediate_shape, final_shape, permutation

def rearrange(tensor: np.ndarray, pattern: str, **named_sizes: Dict[str, int]) -> np.ndarray:
    """
    Rearrange tensor dimensions according to the pattern.
    Arguments:
        tensor (np.ndarray): The input tensor
        pattern (str): The einops pattern to apply
        **named_sizes (Dict[str, int]): Additional named dimensions and their sizes
    Returns:
        np.ndarray: The rearranged tensor
    """
    if '->' not in pattern:
        raise EinopsError("Pattern must contain '->'")
    
    source, target = pattern.split('->')
    source_parsed = ParsedExpression(source.strip())
    target_parsed = ParsedExpression(target.strip())
    
    # Validate dimension counts match tensor shape
    source_dims = source_parsed.actual_dim_count
    if source_parsed.has_ellipsis:
        source_dims += len(tensor.shape) - source_dims
    if source_dims != len(tensor.shape):
        if source_dims < len(tensor.shape):
            raise EinopsError("Pattern requires fewer dimensions")
        else:
            raise EinopsError("Pattern requires more dimensions")
    
    # Get dimension sizes
    try:
        shape_dict = _get_shape_dict(tensor, source_parsed, named_sizes)
    except ValueError as e:
        raise EinopsError(f"Cannot infer sizes: {str(e)}")
    
    target_identifiers = target_parsed.identifiers - {'...'}
    source_identifiers = source_parsed.identifiers - {'...'}
    unknown_dims = target_identifiers - source_identifiers
    if unknown_dims:
        raise EinopsError(f"Unknown dimension(s): {', '.join(unknown_dims)}")
    
    # Compute shapes and permutation
    try:
        intermediate_shape, final_shape, permutation = _compute_output_shape(tensor, target_parsed, source_parsed, shape_dict)
    except KeyError as e:
        raise EinopsError(f"Unknown dimension: {str(e)}")
    
    # Perform the rearrangement
    try:
        if permutation == list(range(len(permutation))):
            return tensor.reshape(final_shape)
        return tensor.reshape(intermediate_shape).transpose(permutation).reshape(final_shape)
    except ValueError as e:
        raise EinopsError(f"Shape Mismatch: {str(e)}")