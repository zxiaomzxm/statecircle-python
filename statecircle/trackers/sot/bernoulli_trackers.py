#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec 26 17:13:44 2019

@author: zhaoxm
"""
import numpy as np
from tqdm import tqdm

from statecircle.trackers.base import SingleObjectTracker
from statecircle.types.state import BernoulliState, GaussianMixtureState, GaussianState
from statecircle.utils.common import IDontKnowHowToDoIt, normalize_log_weights


class BernoulliTracker(SingleObjectTracker):
    """Base Bernoulli tracker class"""

    def __init__(self, prior_birth, surviving_rate, prob_estimate, *args, **kwargs):
        """ Bernoulli RFS defination (Gaussian mixture approximation)
        cyclic Bernoulli state structure
        Bern :
         - prob : float
            existance probability
         - state : `State
            GaussianState(NN or PDA) or GaussianMixtureState(GS)
        """
        super().__init__(*args, **kwargs)
        self.birth_initialized = False

        self.prior_birth = prior_birth
        self.surviving_rate = surviving_rate
        self.prob_estimate = prob_estimate

        # cyclic Bernoulli state in Bernoulli tracker
        self.Bern = BernoulliState(prob=0.)

    def birth(self, meas_data):
        if not self.birth_initialized:
            self.Bern = self.birth_merge(self.birth_model.birth())
            self.birth_initialized = True


class NearestNeighbourBernoulliTracker(BernoulliTracker):
    """NearestNeighbourFilter

    The assumed density is a Bernoulli RFS with a Gaussian density
    """

    def birth_merge(self, birth_state):
        r"""
        Parameters
        ----------
        birth_state : BernoulliState
        """
        max_idx = np.argmax(birth_state.state.log_weights)
        return BernoulliState(prob=birth_state.prob,
                              state=birth_state.state.gaussian_states[max_idx])

    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp

        # Bernoulli state predict
        prob_pred = self.surviving_rate * self.Bern.prob
        prob_birth = self.birth_model.birth().prob * (1 - self.Bern.prob)

        if prob_pred >= prob_birth:
            self.Bern.prob = prob_pred
            self.Bern.state = self.density_model.predict(self.Bern.state, time_step, self.transition_model)
        else:
            # birth step, I can't decouple this part to an independent method because
            # of the 'prob' predict step
            if self.prior_birth:
                self.Bern = self.birth_merge(self.birth_model.birth())
                self.Bern.prob = prob_birth
            else:
                # TODO: see below...
                raise IDontKnowHowToDoIt('How to deal with the clutter measurements?')

    def update(self, meas_data):
        meas = meas_data.meas
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter

        meas_ingate, meas_index = self.gate.gating(self.Bern.state, meas, self.measurement_model)
        pred_loglik = self.density_model.predicted_log_likelihood(self.Bern.state, meas_ingate,
                                                                  self.measurement_model)
        pred_lik = np.exp(pred_loglik)

        # Bernoulli state `prob` update
        delta_t = detection_rate * (1 - np.sum(pred_lik) / intensity_clutter)
        self.Bern.prob = (1 - delta_t) / (1 - self.Bern.prob * delta_t) * self.Bern.prob

        num_meas_ingate = meas_ingate.shape[1]
        log_weights_unnorm = np.empty(num_meas_ingate + 1)
        log_weights_unnorm[0] = np.log(1 - detection_rate)
        log_weights_unnorm[1:] = np.log(detection_rate) + pred_loglik - np.log(intensity_clutter)

        theta_max = np.argmax(log_weights_unnorm)
        if theta_max == 0:
            # missed, not update state
            pass
        else:
            # detected, update Bernoulli state
            self.Bern.state = self.density_model.update(self.Bern.state, meas_ingate[:, theta_max - 1],
                                                        self.measurement_model)

    def estimate(self):
        if self.Bern.prob >= self.prob_estimate:
            return self.estimator(self.Bern.state)

        return np.empty((self.transition_model.ndim, 0))

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # predict
            self.predict(meas_data)

            # birth
            self.birth(meas_data)

            # gating & update
            self.update(meas_data)

            # estimate
            estimates.append(self.estimate())

        return estimates


class ProbabilisticDataAssociationBernoulliTracker(BernoulliTracker):
    """Probabilistic Data Association Bernoulli Tracker

    The assumed density is a Bernoulli RFS with a Gaussian density
    """

    def birth_merge(self, birth_state):
        log_weights, states = birth_state.state.log_weights, birth_state.state.gaussian_states
        return BernoulliState(prob=birth_state.prob,
                              state=self.reductor.moment_matching(log_weights, states))

    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp

        # Bernoulli state predict
        prob_pred = self.surviving_rate * self.Bern.prob
        prob_birth = self.birth_model.birth().prob * (1 - self.Bern.prob)

        if prob_pred >= prob_birth:
            self.Bern.prob = prob_pred
            self.Bern.state = self.density_model.predict(self.Bern.state, time_step, self.transition_model)
        else:
            # birth step, I can't decouple this part to an independent method because
            # of the 'prob' predict step
            if self.prior_birth:
                self.Bern = self.birth_merge(self.birth_model.birth())
                self.Bern.prob = prob_birth
            else:
                # TODO: see below...
                raise IDontKnowHowToDoIt('How to deal with the clutter measurements?')

    def update(self, meas_data):
        meas = meas_data.meas
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter

        # gating
        meas_ingate, meas_index = self.gate.gating(self.Bern.state, meas, self.measurement_model)
        pred_loglik = self.density_model.predicted_log_likelihood(self.Bern.state, meas_ingate, self.measurement_model)
        pred_lik = np.exp(pred_loglik)

        # update
        delta_t = detection_rate * (1 - np.sum(pred_lik) / intensity_clutter)
        self.Bern.prob = (1 - delta_t) / (1 - self.Bern.prob * delta_t) * self.Bern.prob

        num_meas_ingate = meas_ingate.shape[1]
        log_weights_unnorm = np.empty(num_meas_ingate + 1)

        # generate hypothesis tree
        gaussian_states = []

        # missed detection hypothesis
        log_weights_unnorm[0] = np.log(1 - detection_rate)
        gaussian_states.append(self.Bern.state)

        # object detection hypothesis
        log_weights_unnorm[1:] = np.log(detection_rate) + pred_loglik - np.log(intensity_clutter)
        for i in range(1, num_meas_ingate + 1):
            gaussian_states.append(
                self.density_model.update(self.Bern.state, meas_ingate[:, i - 1], self.measurement_model))

        # normalize hypothesis weights
        log_weights, log_sum_weights = normalize_log_weights(log_weights_unnorm)

        # prune hypothesis
        log_weights, gaussian_states = self.reductor.prune(log_weights, gaussian_states)

        # re-normalize
        log_weights, log_sum_w = normalize_log_weights(log_weights)

        # moment matching
        self.Bern.state = self.reductor.moment_matching(log_weights, gaussian_states)

    def estimate(self):
        if self.Bern.prob >= self.prob_estimate:
            return self.estimator(self.Bern.state)

        return np.empty((self.transition_model.ndim, 0))

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # predict
            self.predict(meas_data)

            # birth
            self.birth(meas_data)

            # update
            self.update(meas_data)

            # estimate
            estimates.append(self.estimate())

            # reduction
            self.reduction()

        return estimates


class GaussianSumBernoulliTracker(BernoulliTracker):
    """Gaussian Sum Bernoulli Tracker

    The assumed density is a Bernoulli RFS with a Gaussian mixture density
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: use specified hypothesis type, but here for GST we only use a Bernoulli state
        # TODO: with a Gaussian mixture density. Unlike multi-Bernoulli state can have unnormalized
        # TODO: existence probability, its weights in Gaussian mixture density have to be normalized
        self.Bern = BernoulliState(prob=0., state=GaussianMixtureState())

    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp

        # for each gaussian state in Gaussian mixture, perform prediction
        self.Bern.state.gaussian_states = [self.density_model.predict(state, time_step, self.transition_model)
                                           for state in self.Bern.state.gaussian_states]
        # predict weights
        # TODO: '+=' for empty list '[]'is a bug
        # NOTE: unnormalized log weights
        assert len(self.Bern.state.log_weights) != 0
        self.Bern.state.log_weights += np.log(self.surviving_rate) + np.log(self.Bern.prob)

        # predict existence  probability
        self.Bern.prob = self.surviving_rate * self.Bern.prob

    def birth(self, meas_data):
        meas = meas_data.meas
        # NOTE: we should divide self.Bern.prob by self.surviving_rate to get the 
        # pre-predict prob.
        if self.prior_birth:
            birth_gaussian_mixture = self.birth_model.birth().state
            birth_prob = self.birth_model.birth().prob
            birth_log_weights = birth_gaussian_mixture.log_weights + \
                                np.log(birth_prob) + \
                                np.log(1 - self.Bern.prob / self.surviving_rate)
            log_weights = np.hstack((self.Bern.state.log_weights, birth_log_weights))
            gaussian_states = self.Bern.state.gaussian_states + birth_gaussian_mixture.gaussian_states
            # Normalized log weights in the init method of GaussianMixtureState class
            self.Bern.state = GaussianMixtureState(log_weights, gaussian_states)
            self.Bern.prob += birth_prob * (1 - self.Bern.prob / self.surviving_rate)
        else:
            # TODO: refoctor the birth_model, should have birth_prob, birth_log_weight, 
            # TODO: birth_mean and birth_cov in meas_driven mode
            birth_gaussian_mixture = self.birth_model.birth().state
            birth_prob = self.birth_model.birth().prob
            birth_log_weight = birth_gaussian_mixture.log_weights[0]
            birth_cov = self.birth_model.birth().state.gaussian_states[0].cov
            num_meas = meas.shape[1]
            birth_log_weights = birth_log_weight * np.ones(num_meas) + \
                                np.log(birth_prob) + \
                                np.log(1 - self.Bern.prob / self.surviving_rate)
            log_weights = np.hstack((self.Bern.state.log_weights, birth_log_weights))

            birth_mean = self.measurement_model.reverse(meas)
            gaussian_states = self.Bern.state.gaussian_states + [GaussianState(mean, birth_cov) for mean in
                                                                 birth_mean.T]

            # Normalized log weights in the init method of GaussianMixtureState class
            self.Bern.state = GaussianMixtureState(log_weights, gaussian_states)
            self.Bern.prob += birth_prob * (1 - self.Bern.prob / self.surviving_rate)

    def update(self, meas_data):
        meas = meas_data.meas
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter

        gaussian_states = []
        log_weights = []
        det_weights_sum = 0.
        for log_weight, state in zip(self.Bern.state.log_weights, self.Bern.state.gaussian_states):
            # gating
            meas_ingate, meas_index = self.gate.gating(state, meas, self.measurement_model)
            num_meas_ingate = meas_ingate.shape[1]

            # create missed detection hypothesis for each hypothesis
            gaussian_states.append(state)
            log_weights.append(log_weight + np.log(1 - detection_rate))

            if num_meas_ingate > 0:
                # create object detection hypothesis for each detection inside the gate
                pred_loglik = self.density_model.predicted_log_likelihood(state, meas_ingate, self.measurement_model)
                log_weight_update = log_weight + np.log(detection_rate) + pred_loglik - np.log(intensity_clutter)
                log_weights.append(log_weight_update)
                gaussian_states += [self.density_model.update(state, mea, self.measurement_model)
                                    for mea in meas_ingate.T]
                det_weights_sum += np.sum(np.exp(log_weight_update))

        # normalize hypothesis weights
        log_weights, log_sum_weights = normalize_log_weights(np.hstack(log_weights))

        # update Bernoulli state
        delta_t = detection_rate - det_weights_sum
        self.Bern.prob = (1 - delta_t) / (1 - self.Bern.prob * delta_t) * self.Bern.prob
        self.Bern.state = GaussianMixtureState(log_weights, gaussian_states)

    def estimate(self):
        if self.Bern.prob > self.prob_estimate:
            max_idx = np.argmax(self.Bern.state.log_weights)
            return self.estimator(self.Bern.state.gaussian_states[max_idx])

        return np.empty((self.transition_model.ndim, 0))

    def reduction(self):
        log_weights, states = self.Bern.state.log_weights, self.Bern.state.gaussian_states

        # prune hypothesis with samll weights, and then re-normalize the weights
        log_weights, states = self.reductor.prune(log_weights, states)
        log_weights, log_sum_w = normalize_log_weights(log_weights)

        # hypothesis merging
        log_weights, states = self.reductor.merge(log_weights, states)

        # cap the number of the hypothesis, and then re-normalize the weights
        log_weights, states = self.reductor.cap(log_weights, states)
        log_weights, log_suw_weights = normalize_log_weights(log_weights)

        # return
        self.Bern.state = GaussianMixtureState(log_weights, states)

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # predict
            self.predict(meas_data)

            # birth
            self.birth(meas_data)

            # update
            self.update(meas_data)

            # estimate
            estimates.append(self.estimate())

            # reduction
            self.reduction()

        return estimates
