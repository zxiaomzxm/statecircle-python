#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 25 20:37:07 2019

@author: zhaoxm
"""
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from statecircle.trackers.mot.mbm_filter import MBMFilter
from statecircle.types.state import BernoulliTrajectory, GaussianAccumulatedState
from statecircle.utils.data_association import data_association
from statecircle.utils.common import normalize_log_weights, list_logical_index, list_index
from statecircle.lib.harem.debugger import Debugger
dprint = Debugger.dprint

np.set_printoptions(precision=2)


class MBMTracker(MBMFilter):
    r"""MBM tracker with trajectory state
    """

    def __init__(self, prior_birth, surviving_rate, prob_min, prob_estimate, *args, **kwargs):
        super(MBMTracker, self).__init__(surviving_rate, prob_min, prob_estimate, *args, **kwargs)
        self.prior_birth = prior_birth

    def Bern_predict(self, Bern, time_step):
        """performs prediiction step for a Bernoulli trajectory component
        Parameters
        ----------
        Bern : BernoulliTrajectory

        Notes: inplace prediction
        """
        if Bern.w_death[-1] >= self.prob_min:
            Bern.state = self.density_model.predict(Bern.state, time_step, self.transition_model)
            # predicted time of death
            Bern.t_death = np.append(Bern.t_death, Bern.t_death[-1] + time_step)
            Bern.w_death = np.append(Bern.w_death[:-1],
                                     (Bern.w_death[-1] * (1 - self.surviving_rate),
                                      Bern.w_death[-1] * self.surviving_rate))
        return Bern

    def Bern_undetected_update(self, Bern):
        """caculates the likelihood of missed detection, and creates new local 
        hypotheses due to missed detection.
        Parameters
        ----------
        Bern : BernoulliTrajectory

        Returns
        -------
        BernTrajectory, likelihood
        """
        detection_rate = self.clutter_model.detection_rate
        Bern_undet_upd = deepcopy(Bern)
        l_nodetect = Bern.prob * (1 - detection_rate * Bern.w_death[-1])
        lik_undetected = 1 - Bern.prob + l_nodetect
        Bern_undet_upd.prob = np.exp(np.log(l_nodetect) - np.log(lik_undetected))
        lik_undetected = np.log(lik_undetected)

        # updated time of death
        Bern_undet_upd.w_death = np.append(Bern.w_death[:-1], Bern.w_death[-1] * (1 - detection_rate)) / \
                                 (1 - detection_rate * Bern.w_death[-1])

        return Bern_undet_upd, lik_undetected

    def Bern_detected_update_lik(self, Bern, meas):
        """calculates the predicted likelihood for a given local hypothesis

        Parameters
        ----------
        Bern : BernoulliTrajectory
        meas : ndarray[meas_dim, num_meas]

        Returns
        -------
        detected likelihoods
        """
        lik_detected = self.density_model.predicted_log_likelihood(Bern.state, meas, self.measurement_model) \
                       + np.log(self.clutter_model.detection_rate) + np.log(Bern.prob) + np.log(Bern.w_death[-1]) \
                       - np.log(self.clutter_model.intensity_clutter)
        return lik_detected

    def Bern_detected_update_state(self, Bern, meas):
        """creates the new local hypothesis due to measurement update

        Parameters
        ----------
        Bern : BernoulliTrajectory
        meas : ndarray[meas_dim, num_meas]

        Returns
        -------
        Bern_det_upd
        """

        # creates the new local hypothesis due to measurement update
        # udpated time of death
        Bern_det_upd = BernoulliTrajectory(prob=1.,
                                           state=self.density_model.update(Bern.state, meas, self.measurement_model),
                                           t_birth=Bern.t_birth,
                                           t_death=Bern.t_death[-1:],
                                           w_death=np.array([1.]))

        return Bern_det_upd

    def reduction(self):
        # removes Bernoulli components with small probability of existence, 
        # adds them to the PPP component, and re-index the hypothesis table. If
        # a track contains no single object hypothesis after pruning, this track 
        # is removed.
        prune_threshold = self.prob_min
        num_trees = len(self.hypo_forest)
        for i in range(num_trees):
            # find all Bernoulli components needed to be pruned
            idx = [x.prob < prune_threshold for x in self.hypo_forest[i]]
            if np.any(idx):
                # remove Bernoulli
                # just [self.hypo_forest[i][~idx]
                keep_idx = [not k for k in idx]
                self.hypo_forest[i] = list_logical_index(self.hypo_forest[i], keep_idx)

                # update hypothesis table, if a Bernoulli component is pruned,
                # set its corresponding entry to -1
                idx = np.where(idx)[0]
                for j in range(len(idx)):
                    tmp = self.hypo_table[:, i]
                    tmp[tmp == idx[j]] = -1
                    self.hypo_table[:, i] = tmp

        # remove tracks that contains no valid local hypotheses
        idx = np.any(self.hypo_table != -1, 0)
        self.hypo_table = self.hypo_table[:, idx]
        self.hypo_forest = list_logical_index(self.hypo_forest, idx)
        if 0 in self.hypo_table.shape:
            # ensure the algorithm still works when all Bernoulli are recycled
            self.global_log_weights = []

        # Re-index hypothesis table
        num_trees = len(self.hypo_forest)
        for i in range(num_trees):
            idx = self.hypo_table[:, i] >= 0
            _, self.hypo_table[idx, i] = np.unique(self.hypo_table[idx, i], return_inverse=True)

        # Merge duplicate hypothesis table rows
        if 0 not in self.hypo_table.shape:
            ht, ic = np.unique(self.hypo_table, return_inverse=True, axis=0)
            if ht.shape[0] != self.hypo_table.shape[0]:
                # there are duplicate entries
                w = np.zeros(ht.shape[0])
                for i in range(ht.shape[0]):
                    indices_dupli = ic == i
                    _, w[i] = normalize_log_weights(self.global_log_weights[indices_dupli])
                self.hypo_table = ht
                self.global_log_weights = w

    def predict(self, meas_data):
        """performs MBM prediction step"""
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return

        self.timestamp, time_step = timestamp, timestamp - self.timestamp

        # MB predict
        self.hypo_forest = [[self.Bern_predict(hypo, time_step) for hypo in hypo_tree]
                            for hypo_tree in self.hypo_forest]

    def birth(self, meas_data):
        timestamp, meas = meas_data.timestamp, meas_data.meas
        if self.prior_birth:
            birth_state = self.birth_model.birth().intensity
            # TODO: reformat this birth part! initialize birth density
            # TODO: put this part in the main loop, t_birth = initial_timestamp
            # birth bernoulli components
            self.hypo_forest += [[BernoulliTrajectory(prob, state,
                                                      t_birth=timestamp,
                                                      t_death=np.array([timestamp]),
                                                      w_death=np.array([1]))]
                                 for prob, state in zip(birth_state.weights, birth_state.gaussian_states)]

            # augment look-up table with birth components
            num_global_hypo, num_trees = self.hypo_table.shape
            self.global_log_weights = np.array([0.]) if num_trees == 0 else self.global_log_weights
            ht_birth = np.zeros((num_global_hypo, birth_state.num_comps), dtype=np.int_)
            self.hypo_table = np.hstack((self.hypo_table, ht_birth))
        else:
            # initialize the birth components in the update step if not using prior birth
            # TODO: reformat the birth_model, it shoule has birth_weight and birth_cov property
            # TODO: birth bernoulli components
            num_meas = meas.shape[1]
            birth_weight = self.birth_model.intensity.weights[0]
            birth_cov = self.birth_model.intensity.gaussian_states[0].cov
            for z in meas.T:
                birth_state = GaussianAccumulatedState()
                birth_state.mean = self.measurement_model.reverse(z[:, None])
                birth_state.cov = birth_cov
                self.hypo_forest += [[BernoulliTrajectory(prob=birth_weight,
                                                          state=birth_state,
                                                          t_birth=timestamp,
                                                          t_death=np.array([timestamp]),
                                                          w_death=np.array([1.])
                                                          )
                                      ]]

            # augment look-up table with birth components
            num_global_hypo, num_trees = self.hypo_table.shape
            self.global_log_weights = np.array([0]) if num_trees == 0 else self.global_log_weights
            ht_birth = np.zeros((num_global_hypo, num_meas), dtype=np.int_)
            self.hypo_table = np.hstack((self.hypo_table, ht_birth))

    def update(self, meas_data):
        """performs MBM update step"""
        meas = meas_data.meas
        num_meas = meas.shape[1]

        # 1) Bernoulli update. For each Bernoulli state density_model, create a missed detection
        # hypothesis  (Bernoulli component), and m object detection hypothesis
        # (Bernoulli component), where m is the number of detections inside the 
        # ellipsoidal gate of the giving state density_model
        num_trees = len(self.hypo_forest)
        hypo_forest_update = [None] * num_trees
        loglik_mats = [None] * num_trees
        for i in range(num_trees):
            num_hypos = len(self.hypo_forest[i])
            hypo_forest_update[i] = [None] * (num_hypos * (num_meas + 1))
            loglik_mats[i] = -np.inf * np.ones((num_hypos, num_meas + 1))
            for h in range(num_hypos):
                Bern = self.hypo_forest[i][h]
                # gating
                _, meas_index = self.gate.gating(Bern.state, meas, self.measurement_model)

                idx = h * (num_meas + 1)
                # create a missed detection hypothesis (Bernoulli component)
                Bern_undet, lik_undet = self.Bern_undetected_update(Bern)
                hypo_forest_update[i][idx] = Bern_undet
                loglik_mats[i][h, 0] = lik_undet

                # create a detection hypothesis (Bernoulli component)
                loglik_mats[i][h, [False, *meas_index]] = self.Bern_detected_update_lik(Bern, meas[:, meas_index])
                
                ## hard coding part ###
#                try:
#                    birth_index = np.ix_(np.any(Bern.state.mean[:2]==meas, 0))[0][0]
#                    dprint(loglik_mats[i][h, birth_index + 1])
#                    loglik_mats[i][h, birth_index + 1] = -1.5
#                except:
#                    pass
                #######################
                
                for j in range(num_meas):
                    if meas_index[j]:
                        Bern_det = self.Bern_detected_update_state(Bern, meas[:, j])
                        hypo_forest_update[i][idx + j + 1] = Bern_det
        self.hypo_forest = hypo_forest_update

        # 2) for each global hypothesis, construct the corresponding cost matrix
        # and use Murty's algorithm to obtain the M best global hypothesis with
        # highest weights
        num_global_hypo = len(self.global_log_weights)
        hypo_table = self.hypo_table
        if 0 in hypo_table.shape:
            # initialzie the `hypo_table`
            hypo_table = np.zeros((1, num_meas), dtype=np.int)
            hypo_table_update = hypo_table
        else:
            hypo_table_update = []

        hk = 0
        log_weights_unnorm = []
        for h in range(num_global_hypo):
            # generate cost matrix
            if len(hypo_table) == 0:
                num_trees = 0
            cost_mat = np.inf * np.ones((num_trees, num_meas + num_trees))
            add_back_lik = 0
            for i in range(num_trees):
                if hypo_table[h, i] >= 0:
                    lik_missed = loglik_mats[i][self.hypo_table[h, i], 0]
                    cost_mat[i, num_meas + i] = -lik_missed
                    lik_det = loglik_mats[i][self.hypo_table[h, i], 1:]
                    # NOTE: if `lik_det = some terms - np.log(intensity_c)` in
                    # `self.Bern_detected_update_lik` method, then `lik_det` doesn't
                    # needs minus `np.log(intensity_c)` again here.
                    cost_mat[i, :num_meas] = -lik_det

            if 0 in cost_mat.shape:
                # consider the case that no measurements inside the gate, thus missed detection
                cost = np.array([0])
                col_idx = np.array([[-1]])
                theta_t = col_idx
                max_assoc_num = 0
            else:
                # compute number of assignments
                valid_rows = self.hypo_table[h, :] != -1
                max_assoc_num = np.ceil(np.exp(self.global_log_weights[h]) * self.reductor.capping_num)
                col_idx, cost = data_association(cost_mat[valid_rows, :], max_assoc_num)
                max_assoc_num = col_idx.shape[1]
                theta_t = -1 * np.ones((num_trees, max_assoc_num))
                theta_t[valid_rows, :] = col_idx

            # update global hypo weights
            log_weights_unnorm.append(self.global_log_weights[h] - cost + add_back_lik)

            for m in range(max_assoc_num):
                hk += 1
                theta = theta_t[:, m]

                # 3) update global look-up table
                # TOOD: sanity check for the straight log_weights computing
                # log_weights_unnorm_bak = self.global_log_weights[h]
                table_row = -1 * np.ones(num_trees, dtype=np.int_)
                for i in range(num_trees):
                    if self.hypo_table[h, i] >= 0:
                        assert (theta[i] >= 0)
                        if theta[i] > num_meas - 1:
                            idx = -1
                        else:
                            idx = theta_t[i, m]
                        table_row[i] = self.hypo_table[h, i] * (num_meas + 1) + 1 + idx  # 0 means missed local hypo
                    # increase log_weight
                    # log_weights_unnorm_bak[hk-1] = log_weights_unnorm_bak[hk-1] + assoc_lik[i][hypo_table[h,i], theta[i]]
                hypo_table_update.append(table_row)
                # np.testing.assert_almost_equal(log_weights_unnorm, log_weights_unnorm1)

        hypo_table_update = np.stack(hypo_table_update, 0) if len(hypo_table_update) > 0 else np.empty([1, 0],
                                                                                                       dtype=np.int)

        log_weights_unnorm = np.array([0], dtype=np.float) if len(log_weights_unnorm) == 0 else np.hstack(
            log_weights_unnorm)
        log_weights, log_weights_sum = normalize_log_weights(log_weights_unnorm)

        # 4) prune global hypotheses with small weights and cap the number
        log_weights, keep_idx_prune = self.reductor.prune(log_weights,
                                                          np.arange(len(log_weights)))
        log_weights, keep_idx_cap = self.reductor.cap(log_weights,
                                                      np.arange(len(log_weights)))
        log_weights, log_weights_sum = normalize_log_weights(log_weights)
        hypo_table_update = hypo_table_update[keep_idx_prune, :]
        hypo_table_update = hypo_table_update[keep_idx_cap, :]

        self.global_log_weights = log_weights
        self.hypo_table = hypo_table_update

        # 5) prune local hypotheses (or hypotheses trees) that do not appear in
        # the maintained global hypotheses, and re-index the global hypotheses,
        # and re-index the global hypotheses look-up table
        self.global_log_weights, self.hypo_table, self.hypo_forest = \
            self.prune_hypo_table(self.global_log_weights, self.hypo_table, self.hypo_forest)

    def prune_hypo_table(self, log_w, hypo_table, hypo_forest):
        valid_local_tree_idx = np.any(hypo_table >= 0, 0)
        tt_update = list_logical_index(hypo_forest, valid_local_tree_idx)
        ht_update = hypo_table[:, valid_local_tree_idx]
        for i in range(len(tt_update)):
            local_hypo_idx = ht_update[:, i]
            if np.any(local_hypo_idx == -1):
                local_hypo_idx_nonneg = local_hypo_idx[local_hypo_idx >= 0]
                keep_local_idx, ic = np.unique(local_hypo_idx_nonneg, return_inverse=True)
                tt_update[i] = list_index(tt_update[i], keep_local_idx.astype(np.int))
                tmp = ht_update[:, i]
                tmp[local_hypo_idx >= 0] = ic
                ht_update[:, i] = tmp
            else:
                keep_local_idx, ic = np.unique(local_hypo_idx, return_inverse=True)
                tt_update[i] = list_index(tt_update[i], keep_local_idx.astype(np.int))
                ht_update[:, i] = ic
        log_w_update = log_w

        return log_w_update, ht_update, tt_update

    def estimate(self):
        """performs object state estimation in the MBM tracker"""

        null = [{'range': [-1, -1], 'trajectory': np.empty([self.transition_model.ndim, 0])}]
        if len(self.global_log_weights) == 0:
            return null

        # first, select the multi-Bernoulli with the highest weight
        max_idx = np.argmax(self.global_log_weights)
        local_hypo_idx = self.hypo_table[max_idx, :]

        # second, report the mean of the Bernoulli components whose existence
        # probability is above a threshold
        estimate = []
        for i in range(len(local_hypo_idx)):
            if local_hypo_idx[i] >= 0:
                Bern = self.hypo_forest[i][local_hypo_idx[i]]
                if Bern.prob > self.prob_estimate:
                    ideath = np.argmax(Bern.w_death)
                    t_death = Bern.t_death[ideath]
                    tlen = t_death - Bern.t_birth + 1
                    trajectory = self.estimator(Bern.state)
                    estimate.append({'range': [Bern.t_birth, t_death], 'trajectory': trajectory[:, :tlen]})

        return estimate or null

    def filtering(self, data_reader):
        # tracks multiple objects using multi-Bernoulli mixture filter
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # MBM prediction
            self.predict(meas_data)

            # MBM birth
            self.birth(meas_data)

            # MBM update
            self.update(meas_data)

            # extract state estimates from MBM
            estimates.append(self.estimate())

            # Bern recycling & reduction and PPP reduction
            self.reduction()

        return estimates
