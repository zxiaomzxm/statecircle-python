import numpy as np
from statecircle.types.state import GaussianSumState

def test_gaussian_sum_state():
    state1 = GaussianSumState([1, 2, 3], [1, 2, 3])
    state2 = GaussianSumState([4, 5, 6], [4, 5, 6])
    state_sum = state1 + state2
    state_rsum = state2 + state1
    state_sum_ans = GaussianSumState([1, 2, 3, 4, 5, 6],
                                     [1, 2, 3, 4, 5, 6])
    state_rsum_ans = GaussianSumState([4, 5, 6, 1, 2, 3],
                                     [4, 5, 6, 1, 2, 3])
    np.testing.assert_allclose(state_sum.log_weights, state_sum_ans.log_weights)
    np.testing.assert_allclose(state_sum.gaussian_states, state_sum_ans.gaussian_states)
    np.testing.assert_allclose(state_rsum.log_weights, state_rsum_ans.log_weights)
    np.testing.assert_allclose(state_rsum.gaussian_states, state_rsum_ans.gaussian_states)

    state1 += state1
    np.testing.assert_allclose(state1.log_weights, [1, 2, 3, 1, 2, 3])

