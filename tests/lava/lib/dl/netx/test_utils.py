# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest
import numpy as np
from numpy.testing import assert_array_equal

from lava.lib.dl.netx.utils import optimize_weight_bits, SYNAPSE_SIGN_MODE

class TestOptimizeWeightBits(unittest.TestCase):

    def assertOptimizeWeightBitsEqual(self, actual, expected):
        self.assertTrue(np.all(actual[0] == expected[0]))  # Compare arrays
        self.assertEqual(actual[1:], expected[1:])  # Compare other tuple elements

    def test_4Factor(self):
        weights = np.array([4, 8, 16])
        expected = (np.array([1, 2, 4]), 3, 2, SYNAPSE_SIGN_MODE.EXCITATORY)
        result = optimize_weight_bits(weights)
        self.assertOptimizeWeightBitsEqual(result, expected)

    def test_mixedSignWeights(self):
        weights = np.array([-1, 1])
        expected = (np.array([-1, 1]), 1, 0, SYNAPSE_SIGN_MODE.MIXED)
        result = optimize_weight_bits(weights)
        self.assertOptimizeWeightBitsEqual(result, expected)

    def test_allNegativeWeights(self):
        weights = np.array([-1, -2, -3, -4])
        expected = (np.array([-1, -2, -3, -4]), 2, 0, SYNAPSE_SIGN_MODE.INHIBITORY)
        result = optimize_weight_bits(weights)
        self.assertOptimizeWeightBitsEqual(result, expected)

    def test_weightsBeyondScalingLimits(self):
        weights = np.array([300, 150, -1])
        expected = (np.array([254, 150, -1]), 8, 0, SYNAPSE_SIGN_MODE.MIXED)
        result = optimize_weight_bits(weights)
        self.assertOptimizeWeightBitsEqual(result, expected)

    def test_largeWeightsRange(self):
        weights = np.array([1, -1, 100, -100])
        expected = (np.array([1, -1, 100, -100]), 7, 0, SYNAPSE_SIGN_MODE.MIXED)
        result = optimize_weight_bits(weights)
        self.assertOptimizeWeightBitsEqual(result, expected)
