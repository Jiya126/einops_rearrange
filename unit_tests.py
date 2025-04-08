import unittest
import numpy as np
from typing import List, Tuple

from parser import ParsedExpression, EinopsError
from funcs import rearrange

class TestParsedExpression(unittest.TestCase):
    """Test cases for pattern parsing"""
    
    def test_basic_parsing(self):
        """Test basic pattern parsing without special cases"""
        expr = ParsedExpression('a b c')
        self.assertEqual(expr.composition, [['a'], ['b'], ['c']])
        self.assertEqual(expr.identifiers, {'a', 'b', 'c'})
        self.assertFalse(expr.has_ellipsis)

    def test_grouped_dimensions(self):
        """Test parsing of grouped dimensions"""
        expr = ParsedExpression('a (b c) d')
        self.assertEqual(expr.composition, [['a'], ['b', 'c'], ['d']])
        self.assertEqual(expr.identifiers, {'a', 'b', 'c', 'd'})

    def test_ellipsis(self):
        """Test ellipsis handling"""
        expr = ParsedExpression('a ... c')
        self.assertEqual(expr.composition, [['a'], '_ellipsis_', ['c']])
        self.assertTrue(expr.has_ellipsis)

    def test_invalid_patterns(self):
        """Test various invalid pattern cases"""
        invalid_patterns = [
            'a ((b c)) d',  # Nested parentheses
            'a (b c d',     # Unclosed parenthesis
            'a b) c',       # Unmatched closing parenthesis
            'a ... ... c',  # Multiple ellipsis
            'a (... b) c',  # Ellipsis in parentheses
            '123',          # Invalid identifier
            'a @b c',       # Invalid character
            'a () c',       # Empty parentheses
            '',            # Empty expression
            ' ',           # Whitespace only
        ]
        
        for pattern in invalid_patterns:
            with self.subTest(pattern=pattern):
                with self.assertRaises(EinopsError):
                    ParsedExpression(pattern)

    def test_whitespace_handling(self):
        """Test handling of various whitespace patterns"""
        patterns = [
            ('a  b   c', [['a'], ['b'], ['c']]),
            ('a(b c)d', [['a'], ['b', 'c'], ['d']]),
            (' a b c ', [['a'], ['b'], ['c']]),
        ]
        
        for pattern, expected in patterns:
            with self.subTest(pattern=pattern):
                expr = ParsedExpression(pattern)
                self.assertEqual(expr.composition, expected)

class TestRearrange(unittest.TestCase):
    """Test cases for tensor rearrangement"""
    
    def setUp(self):
        """Set up common test tensors"""
        self.tensor_2d = np.arange(6).reshape(2, 3)
        self.tensor_3d = np.arange(24).reshape(2, 3, 4)
        self.tensor_4d = np.arange(120).reshape(2, 3, 4, 5)

    def assert_shapes_equal(self, tensor: np.ndarray, expected_shape: Tuple[int, ...]):
        """Helper to assert tensor shapes match"""
        self.assertEqual(tensor.shape, expected_shape)

    def test_basic_permutations(self):
        """Test basic dimension permutations"""
        cases = [
            (self.tensor_3d, 'a b c -> b c a', (3, 4, 2)),
            (self.tensor_3d, 'a b c -> c a b', (4, 2, 3)),
            (self.tensor_4d, 'a b c d -> d a b c', (5, 2, 3, 4)),
        ]
        
        for tensor, pattern, expected_shape in cases:
            with self.subTest(pattern=pattern):
                result = rearrange(tensor, pattern)
                self.assert_shapes_equal(result, expected_shape)

    def test_merging_dimensions(self):
        """Test merging of dimensions"""
        cases = [
            (self.tensor_3d, 'a b c -> a (b c)', (2, 12)),
            (self.tensor_4d, 'a b c d -> (a b) (c d)', (6, 20)),
            (self.tensor_4d, 'a b c d -> a (b c d)', (2, 60)),
        ]
        
        for tensor, pattern, expected_shape in cases:
            with self.subTest(pattern=pattern):
                result = rearrange(tensor, pattern)
                self.assert_shapes_equal(result, expected_shape)

    def test_splitting_dimensions(self):
        """Test splitting of dimensions"""
        tensor = np.zeros((4, 6))
        cases = [
            ('(a b) c -> a b c', {'a': 2}, (2, 2, 6)),
            ('a (b c) -> a b c', {'b': 2}, (4, 2, 3)),
        ]
        
        for pattern, sizes, expected_shape in cases:
            with self.subTest(pattern=pattern):
                result = rearrange(tensor, pattern, **sizes)
                self.assert_shapes_equal(result, expected_shape)

    def test_error_cases(self):
        """Test various error conditions"""
        tensor = np.zeros((2, 3, 4))
        error_cases = [
            ('a b -> a b c', {}),
            ('a b c d -> a b c', {}),
            ('(a b) c -> a b c', {}),
            ('a ... ... -> a', {}),
            ('a b -> b a c', {}),
            ('a b c -> a b d', {}),
            ('a b -> (a b) c', {}),
            ('a b c -> a (b d)', {}),
            ('', {}),
            ('a b c -> ', {}),
            ('-> a b c', {}),
            ('a b c', {}),
        ]
        
        for pattern, sizes in error_cases:
            with self.subTest(pattern=pattern):
                with self.assertRaises(EinopsError):
                    rearrange(tensor, pattern, **sizes)

    def test_dimension_inference(self):
        """Test automatic dimension size inference"""
        tensor = np.zeros((6, 8))
        result = rearrange(tensor, '(a b) c -> a b c', b=2)
        self.assert_shapes_equal(result, (3, 2, 8))
        
        with self.assertRaises(EinopsError):
            rearrange(tensor, '(a b) c -> a b c')

    def test_inconsistent_sizes(self):
        """Test handling of inconsistent dimension sizes"""
        tensor = np.zeros((4, 4))
        with self.assertRaises(EinopsError):
            rearrange(tensor, 'a a -> a a', a=2)

    def test_shape_mismatch(self):
        """Test handling of shape mismatches"""
        tensor = np.zeros((4, 4))
        with self.assertRaises(EinopsError):
            rearrange(tensor, '(a b) c -> a b c', a=3)

    def test_value_preservation(self):
        """Test that values are correctly preserved after rearrangement"""
        tensor = np.arange(24).reshape(2, 3, 4)
        result = rearrange(tensor, 'a b c -> c b a')
        np.testing.assert_array_equal(
            tensor[0, 1, 2],
            result[2, 1, 0]
        )

# Run the tests directly
if __name__ == '__main__':
    unittest.main(verbosity=2)