#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:06:01 2019

@author: zhxm
"""

from murty import Murty
import numpy as np

# TODO: fix the bug in murty cpp file which the cost matrix has the max negative number
# TODO: This case can happen in the configuration which clutter_intensity == 0
def data_association(cost_mat, topN):
    # TODO: temp workaround
    cost_mat[np.isinf(cost_mat)] = 10000
    # TODO: one way to walk around the annother annoy bug which two rows' value are same in cost matrix
    cost_mat += 1e-10 * np.random.randn(*cost_mat.shape)
    op = Murty(cost_mat)
    status = True
    cost = []
    col_idx = []
    num = 0
    while status and num < topN:
        status, cost_iter, col_iter = op.draw()
        if status:
            cost.append(cost_iter)
            col_idx.append(col_iter)
            num += 1
    assert len(cost) > 0, "invalid cost matrix."
    cost = np.array(cost)
    col_idx = np.stack(col_idx, -1)
    return col_idx, cost 


if __name__ == "__main__":
    cost_mat = np.eye(4)
    col_idx, cost = data_association(cost_mat, 20)
    print(col_idx)
    print(col_idx.shape)
    print(cost)
        
        
