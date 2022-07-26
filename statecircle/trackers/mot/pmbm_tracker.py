#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 25 09:47:52 2019

@author: zhaoxm
"""
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from statecircle.datasets.base import SENSOR_TYPE
from statecircle.trackers.mot.pmbm_filter import PMBMFilter
from statecircle.types.state import BernoulliTrajectory, GaussianState
from statecircle.utils.data_association import data_association
from statecircle.utils.common import normalize_log_weights, list_logical_index, list_index
from statecircle.lib.harem.debugger import Debugger
dprint = Debugger.dprint

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.set_printoptions(precision=2)

class PMBMTracker(PMBMFilter):
    r"""PMBM tracker with trajectory state
    """

    def __init__(self, prior_birth, surviving_rate, recycle_threshold, prob_min, prob_estimate, 
                 meas_models_dict, clutter_models_dict, *args, **kwargs):
        super(PMBMTracker, self).__init__(surviving_rate, recycle_threshold, prob_min, prob_estimate, *args, **kwargs)
        self.prior_birth = prior_birth
        self.meas_models_dict = meas_models_dict
        self.clutter_models_dict = clutter_models_dict

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
            Bern.t_death = np.append(Bern.t_death, Bern.t_death[-1] + 1)
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
        lik_detected = self.density_model.predicted_log_likelihood(Bern.state, meas, self.measurement_model) + \
                       np.log(self.clutter_model.detection_rate) + np.log(Bern.prob) + np.log(Bern.w_death[-1])
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

    def PPP_predict(self, time_step):
        """performs prediction step for PPP components hypothesising undetected 
           objects
        """
        self.ppp.intensity.log_weights = self.ppp.intensity.log_weights + np.log(self.surviving_rate)
        self.ppp.intensity.gaussian_states = [self.density_model.predict_state(state, time_step, self.transition_model)
                                              for state in self.ppp.intensity.gaussian_states]

    def PPP_birth(self, meas_data):
        if self.prior_birth: # prior birth model
            birth_state = self.birth_model.birth().intensity
            self.ppp.intensity += deepcopy(birth_state)
        else: # measurement driven birth model
            # check the sensor_type if meas_data has the attribute,
            # We don't birth object with the radar detections
            if hasattr(meas_data, 'sensor_type') and meas_data.sensor_type in SENSOR_TYPE.RADAR_TYPES:
                return
            
            # initialize the birth components in the update step if using measurements driven strategy
            # TODO: reformat the birth_model, it shoule has birth_weight and birth_cov property
            meas = meas_data.meas
            num_meas = meas.shape[1]
            birth_weight = self.birth_model.intensity.log_weights[0]
            birth_cov = self.birth_model.intensity.gaussian_states[0].cov
            self.ppp.intensity.log_weights = np.hstack((self.ppp.intensity.log_weights,
                                                        [birth_weight] * num_meas))
            for i in range(num_meas):
                # TODO: reformat GaussianState to GaussianAccumulatedState, and predict_state to predict method
                # TODO: in density_model
                birth_mean = self.measurement_model.reverse(meas[:, i:i+1])
                birth_state = GaussianState(birth_mean, birth_cov)
                self.ppp.intensity.gaussian_states.append(birth_state)

    def PPP_detected_update(self, step, indices, mea):
        r"""creates a new local hypothesis by updating the PPP with measurement and
           calculates the corresponding likelihood

        Parameters
        ----------
        step : scalar
            iter time step
        indices : bool matrix
            represent which ppp component contains the measurement
        mea : ndarray[meas_dim, 1]

        Returns
        -------
        Bern_upd, likelihood
        """
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter

        # 1) for each mixture component in the PPP intensity, perform Kalman update and
        # calculate the predicted likelihood for each detection inside the corresponding
        # gate
        states = [state for i, state in enumerate(self.ppp.intensity.gaussian_states) if indices[i]]
        log_w = [np.log(detection_rate) + self.ppp.intensity.log_weights[i] +
                 self.density_model.predicted_log_likelihood(state, mea, self.measurement_model)
                 for i, state in enumerate(self.ppp.intensity.gaussian_states) if indices[i]]

        log_w, log_sum_w = normalize_log_weights(log_w)
        rho = np.exp(log_sum_w)

        # 2) perform Gaussian moment matching for the updated object state
        # densities resulted from being updated by the same detection
        Bern = BernoulliTrajectory()
        Bern.state = self.reductor.moment_matching(log_w, states)
        if np.any(indices):
            # 3) the returned likelihood should be the sum of the predicted
            # likelihoods calculated for each mixture component in the PPP
            # intensity and the clutter intensity
            lik_new = np.log(intensity_clutter + rho)

            # 4) the return existence probability of the Bernoulli component
            # is the ratio between the sum of the predicted likelihoods and
            # the returned likelihood
            Bern.prob = rho / (intensity_clutter + rho)
        else:
            lik_new = np.log(intensity_clutter)
        Bern.t_birth = step
        Bern.t_death = np.array([step])
        Bern.w_death = np.array([1.])

        return Bern, lik_new

    def PPP_undetected_update(self):
        """perform PPP update for missed detection"""
        self.ppp.intensity.log_weights = np.log(1 - self.clutter_model.detection_rate) + \
                                         self.ppp.intensity.log_weights

    def PPP_reduction(self):
        # prunes mixture components in the PPP intensity by pruning
        self.ppp.intensity.log_weights, self.ppp.intensity.gaussian_states = self.reductor.prune(
            self.ppp.intensity.log_weights,
            self.ppp.intensity.gaussian_states)
        self.ppp.intensity.log_weights, self.ppp.intensity.gaussian_states = self.reductor.merge(
            self.ppp.intensity.log_weights,
            self.ppp.intensity.gaussian_states)

    def Bern_reduction(self):
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

    def reduction(self):
        self.Bern_reduction()
        self.PPP_reduction()

    def predict(self, meas_data):
        """performs PMBM prediction step"""
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp

        # PPP predict
        self.PPP_predict(time_step)

        # MB predict
        self.hypo_forest = [[self.Bern_predict(hypo, time_step) for hypo in hypo_tree]
                            for hypo_tree in self.hypo_forest]
        
    def birth(self, meas_data):
        # pre-birth hook
        if hasattr(meas_data, 'sensor_type'):
            # update measurement model
            self.measurement_model = self.meas_models_dict[meas_data.sensor_type]
            self.clutter_model = self.clutter_models_dict[meas_data.sensor_type]
            
        self.PPP_birth(meas_data)

    def update(self, step, meas_data):
        """performs PMBM update step"""
        timestamp, meas = meas_data.timestamp, meas_data.meas
        num_meas = meas.shape[1]

        # 1) perform ellipsoidal gating for each Bernoulli state density_model
        # and each mixture component in the PPP intensity
        num_birth = len(self.ppp.intensity.log_weights)
        meas_birth_indices = np.zeros((num_birth, num_meas))
        for i, state in enumerate(self.ppp.intensity.gaussian_states):
            _, meas_index = self.gate.gating(state, meas, self.measurement_model)
            meas_birth_indices[i, :] = meas_index
        meas_ingate_birth = np.any(meas_birth_indices, 0)

        # 2) Bernoulli update. For each Bernoulli state density_model, create a missed detection
        # hypothesis  (Bernoulli component), and m object detection hypothesis
        # (Bernoulli component), where m is the number of detections inside the 
        # ellipsoidal gate of the giving state density_model
        num_trees = len(self.hypo_forest)
        hypo_forest_update = [None] * num_trees
        assoc_loglik_mats = [None] * num_trees
        for i in range(num_trees):
            num_hypos = len(self.hypo_forest[i])
            hypo_forest_update[i] = [None] * (num_hypos * (num_meas + 1))
            assoc_loglik_mats[i] = -np.inf * np.ones((num_hypos, num_meas + 1))
            for h in range(num_hypos):
                Bern = self.hypo_forest[i][h]
                # gating
                _, meas_index = self.gate.gating(Bern.state, meas, self.measurement_model)

                idx = h * (num_meas + 1)
                # create a missed detection hypothesis (Bernoulli component)
                Bern_undet, lik_undet = self.Bern_undetected_update(Bern)
                hypo_forest_update[i][idx] = Bern_undet
                assoc_loglik_mats[i][h, 0] = lik_undet

                # create a detection hypothesis (Bernoulli component)
                assoc_loglik_mats[i][h, [False, *meas_index]] = self.Bern_detected_update_lik(Bern, meas[:, meas_index])
                for j in range(num_meas):
                    if meas_index[j]:
                        Bern_det = self.Bern_detected_update_state(Bern, meas[:, j])
                        hypo_forest_update[i][idx + j + 1] = Bern_det

        # 3) update PPP with detections. Note that for detection that are not
        # inside the gate of undetected objects, create dummy Bernoulli components
        # withe existence probability r = 0, in this case, the corresponding likelihood
        # is simply clutter intensity
        assoc_loglik_birth = -np.inf * np.ones(num_meas)
        hypo_forest_birth_update = [None] * num_meas
        Bern_dummy = BernoulliTrajectory()
        for j in range(num_meas):
            indices = meas_birth_indices[:, j]
            if np.any(indices):
                Bern, lik_new = self.PPP_detected_update(step, indices, meas[:, j])
            else:
                Bern, lik_new = Bern_dummy, np.log(self.clutter_model.intensity_clutter)
            hypo_forest_birth_update[j] = [Bern]  # new born leaf
            assoc_loglik_birth[j] = lik_new
        hypo_forest_birth_update = [hypo_forest_birth_update[i] for i in range(len(hypo_forest_birth_update)) if
                                    meas_ingate_birth[i]]
        birth_cost_mat = np.diag(-assoc_loglik_birth)
        birth_cost_mat[birth_cost_mat == 0] = np.inf
        self.hypo_forest = hypo_forest_update + hypo_forest_birth_update

        # 4) for each global hypothesis, construct the corresponding cost matrix
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
            cost_mat = np.inf * np.ones((num_meas, num_meas + num_trees))
            add_back_lik = 0
            for i in range(num_trees):
                if hypo_table[h, i] >= 0:
                    lik_det = assoc_loglik_mats[i][hypo_table[h, i], 1:]
                    lik_missed = assoc_loglik_mats[i][hypo_table[h, i], 0]
                    add_back_lik += lik_missed
                    cost_mat[:, i] = -(lik_det - lik_missed)
            cost_mat[:num_meas, num_trees:num_trees + num_meas] = birth_cost_mat

            if 0 in cost_mat.shape:
                # consider the case that no measurements inside the gate, thus missed detection
                cost = np.array([0])
                col_idx = np.array([[-1]])
            else:
                # compute number of assignments
                max_assoc_num = np.ceil(np.exp(self.global_log_weights[h]) * self.reductor.capping_num)
                col_idx, cost = data_association(cost_mat, max_assoc_num)
            psi_m = col_idx

            # update global hypo weights
            log_weights_unnorm.append(self.global_log_weights[h] - cost + add_back_lik)

            for m in range(psi_m.shape[1]):
                hk += 1
                psi = psi_m[:, m]

                # 5) update global look-up table
                table_row = -1 * np.ones(num_trees + num_meas, dtype=np.int)
                for i in range(num_trees):
                    if hypo_table[h, i] >= 0:
                        # find the idx detection associated to obj i
                        idx = np.where(psi == i)[0]
                        if len(idx) == 0:
                            # missed hypothesis index equals 0
                            idx = -1
                        # idx + 1 means the first hypothesis is missed hypothesis,
                        # thus, idx = -1 => idx + 1 = 0 stands for missed hypothesis
                        # and idx >=1 stand for detected hypotheses.
                        table_row[i] = hypo_table[h, i] * (num_meas + 1) + idx + 1

                for i in range(num_trees, num_trees + num_meas):
                    # find the idx detection associated to obj i
                    idx = np.where(psi == i)[0]
                    if len(idx) > 0:
                        table_row[i] = 0

                hypo_table_update.append(table_row)
        hypo_table_update = np.stack(hypo_table_update, 0) if len(hypo_table_update) > 0 else np.empty([1, 0],
                                                                                                       dtype=np.int)

        log_weights_unnorm = np.array([0], dtype=np.float) if len(log_weights_unnorm) == 0 else np.hstack(
            log_weights_unnorm)
        log_weights, log_weights_sum = normalize_log_weights(log_weights_unnorm)

        # 6) update PPP intensity wit missed detection
        self.PPP_undetected_update()

        # 7) prune global hypotheses with small weights and cap the number
        log_weights, keep_idx_prune = self.reductor.prune(log_weights,
                                                          np.arange(len(log_weights)))
        log_weights, keep_idx_cap = self.reductor.cap(log_weights,
                                                      np.arange(len(log_weights)))
        log_weights, log_weights_sum = normalize_log_weights(log_weights)
        hypo_table_update = hypo_table_update[keep_idx_prune, :]
        hypo_table_update = hypo_table_update[keep_idx_cap, :]

        self.global_log_weights = log_weights
        valid_col = np.bool_(np.concatenate((np.ones(num_trees, dtype=np.bool), meas_ingate_birth)))
        self.hypo_table = hypo_table_update[:, valid_col] if np.any(valid_col) else np.empty([1, 0], dtype=np.int)

        # 8) prune local hypotheses (or hypotheses trees) that do not appear in
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
        """performs object state estimation in the PMBM tracker"""

        null = [{'range': [-1, -1], 'trajectory': np.empty([self.transition_model.ndim, 0])}]
        if len(self.global_log_weights) == 0:
            return null

        # first, select the multi-Bernoulli with the highest weight
        max_idx = np.argmax(self.global_log_weights)
        local_hypo_idx = self.hypo_table[max_idx, :]
#        dprint(self.hypo_table.shape)
        # second, report the mean of the Bernoulli components whose existence
        # probability is above a threshold
        estimate = []
        for i in range(len(local_hypo_idx)):
            if local_hypo_idx[i] >= 0:
                Bern = self.hypo_forest[i][local_hypo_idx[i]]
                if Bern.prob > self.prob_estimate:
                    ideath = np.argmax(Bern.w_death)
                    t_death = Bern.t_death[ideath]
                    # TODO: fix this
                    tlen = t_death - Bern.t_birth + 1
                    trajectory = self.estimator(Bern.state)
                    estimate.append({'range': [Bern.t_birth, t_death], 
                                     'trajectory': trajectory[:, :tlen],
                                     'cov': Bern.state.cov[-Bern.ndim:, -Bern.ndim:]})

        return estimate or null

    def filtering(self, data_reader):
        # tracks multiple objects using Poisson multi-Bernoulli mixture filter
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # PMBM prediction
            self.predict(meas_data)
            
            # PMBM birth
            self.birth(meas_data)

            # PMBM update
            self.update(t, meas_data)

            # extract state estimates from PMBM
            estimates.append(self.estimate())

            # Bern recycling & reduction and PPP reduction
            self.reduction()
            
            ############################
            # tmp code for debug info
            ############################
            """ 
            dprint(self.hypo_table.shape)
            dprint(self.hypo_table)
            try:
                dprint(self.hypo_forest[0][0].prob)
                dprint(self.hypo_forest[0][0].t_birth)
                dprint(self.hypo_forest[0][0].t_death)
            except:
                pass
            for i, hypo_tree in enumerate(self.hypo_forest):
                print('tree: {}, #leaves: {}'.format(i, len(hypo_tree)))
            leaves = [len(hypo_tree) for i, hypo_tree in enumerate(self.hypo_forest)]
            tree_probs = [sum([leaf.prob for leaf in hypo_tree]) for i, hypo_tree in enumerate(self.hypo_forest)]
            dprint(np.array([leaves, tree_probs]))
            tree_leaf = dict(zip(list(range(len(leaves))), leaves))
            """

        return estimates
