import json
import os
import time
import numpy as np

from .base import BaseExperiment, OUTPUT_DIR

import solvers


# Constants (default values unless provided by caller)
MAX_STEPS = 2000
NUM_TRIALS = 100
DISCOUNTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
THETA = 0.0001


VI_DIR = os.path.join(OUTPUT_DIR, 'VI')
if not os.path.exists(VI_DIR):
    os.makedirs(VI_DIR)
PKL_DIR = os.path.join(VI_DIR, 'pkl')
if not os.path.exists(PKL_DIR):
    os.makedirs(PKL_DIR)
IMG_DIR = os.path.join(os.path.join(OUTPUT_DIR, 'images'), 'VI')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)


class ValueIterationExperiment(BaseExperiment):

    def __init__(self, details, verbose=False, max_steps = MAX_STEPS, num_trials = NUM_TRIALS, theta = THETA,
                 discounts = DISCOUNTS):
        super(ValueIterationExperiment, self).__init__(details, verbose, max_steps)
        self._num_trials = num_trials
        self._theta = theta
        self._discount_factors = discounts

    def convergence_check_fn(self, solver, step_count):
        return solver.has_converged()

    def perform(self):
        # Value iteration
        self._details.env.reset()
        map_desc = self._details.env.unwrapped.desc

        grid_file_name = os.path.join(VI_DIR, '{}_grid.csv'.format(self._details.env_name))
        with open(grid_file_name, 'w') as f:
            f.write("params,time,steps,reward_mean,reward_median,reward_min,reward_max,reward_std\n")

        dims = len(self._discount_factors)
        self.log("Searching VI in {} dimensions".format(dims))

        runs = 1
        for discount_factor in self._discount_factors:
            t = time.clock()
            self.log("{}/{} Processing VI with discount factor {}".format(runs, dims, discount_factor))

            v = solvers.ValueIterationSolver(self._details.env, discount_factor=discount_factor, theta=self._theta)

            stats = self.run_solver_and_collect(v, self.convergence_check_fn)

            self.log("Took {} steps".format(len(stats.steps)))
            stats.to_csv(os.path.join(VI_DIR, '{}_{}.csv'.format(self._details.env_name, discount_factor)))
            stats.pickle_results(os.path.join(PKL_DIR, '{}_{}_{}.pkl'.format(self._details.env_name, discount_factor, '{}')), map_desc.shape)
            stats.plot_policies_on_map(os.path.join(IMG_DIR, '{}_{}_{}.png'.format(self._details.env_name, discount_factor, '{}_{}')),
                                       map_desc, self._details.env.colors(), self._details.env.directions(),
                                       'Value Iteration', 'Step', self._details, only_last=True)

            optimal_policy_stats = self.run_policy_and_collect(v, stats.optimal_policy, self._num_trials)
            self.log('{}'.format(optimal_policy_stats))
            optimal_policy_stats.to_csv(os.path.join(VI_DIR, '{}_{}_optimal.csv'.format(self._details.env_name, discount_factor)))
            with open(grid_file_name, 'a') as f:
                f.write('"{}",{},{},{},{},{},{},{}\n'.format(
                    json.dumps({'discount_factor': discount_factor}).replace('"', '""'),
                    time.clock() - t,
                    len(optimal_policy_stats.rewards),
                    optimal_policy_stats.reward_mean,
                    optimal_policy_stats.reward_median,
                    optimal_policy_stats.reward_min,
                    optimal_policy_stats.reward_max,
                    optimal_policy_stats.reward_std,
                ))
            runs += 1

