import torch
import numpy as np

from models.positional_encoding import PositionalEncoding


def test_single_batch_no_concat():
    """Test `test_single_batch_concat` with concat_inputs flag not set"""
    input_values = torch.tensor([
        [1, -1],
        [2, -2]
    ], dtype=torch.float32)
    concat_inputs = False
    frequency_count = 3

    positional_encoding = PositionalEncoding(frequency_count=frequency_count, concat_inputs=concat_inputs)

    expected_output_values = torch.tensor([
        [
            np.sin(1),
            np.sin(-1),
            np.sin(2),
            np.sin(-2),
            np.sin(4),
            np.sin(-4),
            np.cos(1),
            np.cos(-1),
            np.cos(2),
            np.cos(-2),
            np.cos(4),
            np.cos(-4)
        ],
        [
            np.sin(2),
            np.sin(-2),
            np.sin(4),
            np.sin(-4),
            np.sin(8),
            np.sin(-8),
            np.cos(2),
            np.cos(-2),
            np.cos(4),
            np.cos(-4),
            np.cos(8),
            np.cos(-8)
        ],
    ], dtype=torch.float32)
    print(positional_encoding(input_values).shape)
    assert torch.allclose(positional_encoding(input_values), expected_output_values)


def test_single_batch_concat():
    """Test `test_single_batch_concat` with concat_inputs flag set"""
    input_values = torch.tensor([
        [1, -1],
        [2, -2]
    ], dtype=torch.float32)
    concat_inputs = True
    frequency_count = 3

    positional_encoding = PositionalEncoding(frequency_count=frequency_count, concat_inputs=concat_inputs)

    expected_output_values = torch.tensor([
        [
            1,
            -1,
            np.sin(1),
            np.sin(-1),
            np.sin(2),
            np.sin(-2),
            np.sin(4),
            np.sin(-4),
            np.cos(1),
            np.cos(-1),
            np.cos(2),
            np.cos(-2),
            np.cos(4),
            np.cos(-4)
        ],
        [
            2,
            -2,
            np.sin(2),
            np.sin(-2),
            np.sin(4),
            np.sin(-4),
            np.sin(8),
            np.sin(-8),
            np.cos(2),
            np.cos(-2),
            np.cos(4),
            np.cos(-4),
            np.cos(8),
            np.cos(-8)
        ],
    ], dtype=torch.float32)
    print(positional_encoding(input_values).shape)
    assert torch.allclose(positional_encoding(input_values), expected_output_values)



def test_batch_no_concat():
    """Test `test_single_batch_concat` with concat_inputs flag set"""
    input_values = torch.tensor([
        [[1, -1]],
        [[2, -2]]
    ], dtype=torch.float32)
    concat_inputs = True
    frequency_count = 2

    positional_encoding = PositionalEncoding(frequency_count=frequency_count, concat_inputs=concat_inputs)

    expected_output_values = torch.tensor([
        [
            [
                1,
                -1,
                np.sin(1),
                np.sin(-1),
                np.sin(2),
                np.sin(-2),
                np.cos(1),
                np.cos(-1),
                np.cos(2),
                np.cos(-2),
            ],
        ],
        [
            [
                2,
                -2,
                np.sin(2),
                np.sin(-2),
                np.sin(4),
                np.sin(-4),
                np.cos(2),
                np.cos(-2),
                np.cos(4),
                np.cos(-4),
            ],
        ]
    ], dtype=torch.float32)
    print(positional_encoding(input_values).shape)
    assert torch.allclose(positional_encoding(input_values), expected_output_values)
