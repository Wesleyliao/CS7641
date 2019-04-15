import argparse
from datetime import datetime
import logging
import random as rand
import numpy as np

import environments
import experiments
from experiments import plotting

# Get parameters from external file (./parameters.py) if provided
try:
    import parameters
    ENV_REWARDS = parameters.ENV_REWARDS
    MAX_STEPS = parameters.MAX_STEPS
    NUM_TRIALS = parameters.NUM_TRIALS
    PI_THETA = parameters.PI_THETA
    VI_THETA = parameters.VI_THETA
    QL_THETA = parameters.QL_THETA
    PI_DISCOUNTS = parameters.PI_DISCOUNTS
    VI_DISCOUNTS = parameters.VI_DISCOUNTS
    QL_DISCOUNTS = parameters.QL_DISCOUNTS
    QL_MAX_EPISODES = parameters.QL_MAX_EPISODES
    QL_MIN_EPISODES = parameters.QL_MIN_EPISODES
    QL_MAX_EPISODE_STEPS = parameters.QL_MAX_EPISODE_STEPS
    QL_MIN_SUB_THETAS = parameters.QL_MIN_SUB_THETAS
    QL_ALPHAS = parameters.QL_ALPHAS
    QL_Q_INITS = parameters.QL_Q_INITS
    QL_EPSILONS = parameters.QL_EPSILONS
    QL_EPSILON_DECAYS = parameters.QL_EPSILON_DECAYS
    IMPORTED = True
except:
    IMPORTED = False


if not IMPORTED:

    # THE FOLLOWING CONFIGURATION PARAMETERS MUST ALL BE SET

    # Configure rewards per environment
    ENV_REWARDS = {
                   'small_lake':    { 'step_prob': 0.8, # Float
                                      'step_rew': -0.1, # Float
                                      'hole_rew': -1, # Float
                                      'goal_rew': 10, # Float
                                    },
                   'large_lake':    { 'step_prob': 0.8, # Float
                                      'step_rew': -0.1, # Float
                                      'hole_rew': -1, # Float
                                      'goal_rew': 10, # Float
                                    },
                   'cliff_walking': { 'wind_prob': 0.2, # Float
                                      'step_rew': -0.1, # Float
                                      'fall_rew': -1, # Float
                                      'goal_rew': 10, # Float
                                    },
                  }

    # Configure max steps per experiment
    MAX_STEPS = { 'pi': 100, # Int
                  'vi': 100, # Int
                  'ql': 200, # Int
                }

    # Configure trials per experiment
    NUM_TRIALS = { 'pi': 100, # Int
                   'vi': 100, # Int
                   'ql': 200, # Int
                 }

    # Configure thetas per experiment
    PI_THETA = 0.00001 # Float
    VI_THETA = 0.00001 # Float
    QL_THETA = 0.00001 # Float

    # Configure discounts per experiment (lists of discount values)
    PI_DISCOUNTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # List of floats
    VI_DISCOUNTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # List of floats
    QL_DISCOUNTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # List of floats

    # Configure other QL experiment parameters
    QL_MAX_EPISODES = 200 # Int
    QL_MIN_EPISODES = 5 # Int
    QL_MAX_EPISODE_STEPS = 200 # Int
    QL_MIN_SUB_THETAS = 5 # (Int) number of consecutive episodes with little change before calling it converged
    QL_ALPHAS = [0.1, 0.5, 0.9] # List of floats
    QL_Q_INITS = ['random', 0,] # (the string 'random' or floats) a list of q-inits to try
    QL_EPSILONS = [0.1, 0.3, 0.5] # List of floats
    QL_EPSILON_DECAYS = [0.0001] # a list of floats


# Check configuration settings (just make sure they've been set to something)
for env in ENV_REWARDS.keys():
    for setting in ENV_REWARDS[env].keys():
        if ENV_REWARDS[env][setting] is None:
            print(env + ' ' + setting + ' not set!')
            quit()
for exp in MAX_STEPS.keys():
    if MAX_STEPS[exp] is None:
        print(exp.upper() + ' max steps not set!')
        quit()
for exp in NUM_TRIALS.keys():
    if NUM_TRIALS[exp] is None:
        print(exp.upper() + ' num trials not set!')
        quit()
if PI_THETA is None or VI_THETA is None or QL_THETA is None:
    print('Not all experiment thetas set!')
    quit()
if len(PI_DISCOUNTS) == 0 or len(VI_DISCOUNTS) == 0 or len(QL_DISCOUNTS) == 0:
    print('Not all experiment discounts set!')
    quit()
if QL_MAX_EPISODES is None or QL_MIN_EPISODES is None or QL_MAX_EPISODE_STEPS is None or QL_MIN_SUB_THETAS is None \
   or len(QL_ALPHAS) == 0 or len(QL_Q_INITS) == 0 or len(QL_EPSILONS) == 0 or len(QL_EPSILON_DECAYS) == 0:
    print('Not all QL experiment parameters set!')
    quit()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, verbose, timings, max_steps, num_trials, \
                   theta = None, max_episodes = None, min_episodes = None, max_episode_steps = None, \
                   min_sub_thetas = None, discounts = None, alphas = None, q_inits = None, epsilons = None, \
                   epsilon_decays = None):

    timings[timing_key] = {}
    for details in experiment_details:
        t = datetime.now()
        logger.info("Running {} experiment: {}".format(timing_key, details.env_readable_name))
        if timing_key == 'QL': # Q-Learning
            exp = experiment(details, verbose=verbose, max_steps=max_steps, num_trials=num_trials,
                             max_episodes=max_episodes, min_episodes=min_episodes, max_episode_steps=max_episode_steps,
                             min_sub_thetas=min_sub_thetas, theta=theta, discounts=discounts, alphas=alphas,
                             q_inits=q_inits, epsilons=epsilons, epsilon_decays=epsilon_decays)
        else: # NOT Q-Learning
            exp = experiment(details, verbose=verbose, max_steps=max_steps, num_trials=num_trials, theta=theta,
                             discounts=discounts)
        exp.perform()
        t_d = datetime.now() - t
        timings[timing_key][details.env_name] = t_d.seconds


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run MDP experiments')
    parser.add_argument('--threads', type=int, default=-1, help='Number of threads (defaults to -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--policy', action='store_true', help='Run the Policy Iteration (PI) experiment')
    parser.add_argument('--value', action='store_true', help='Run the Value Iteration (VI) experiment')
    parser.add_argument('--ql', action='store_true', help='Run the Q-Learner (QL) experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    # Set random seed
    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 31) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Creating MDPs")
    logger.info("----------")

    # Modify this list of dicts to add/remove/swap environments
    envs = [
        {
            'env': environments.get_rewarding_frozen_lake_8x8_environment(ENV_REWARDS['small_lake']['step_prob'],
                                                                          ENV_REWARDS['small_lake']['step_rew'],
                                                                          ENV_REWARDS['small_lake']['hole_rew'],
                                                                          ENV_REWARDS['small_lake']['goal_rew']),
            'name': 'frozen_lake',
            'readable_name': 'Frozen Lake (8x8)',
        },
        {
            'env': environments.get_large_rewarding_frozen_lake_15x15_environment(ENV_REWARDS['large_lake']['step_prob'],
                                                                                  ENV_REWARDS['large_lake']['step_rew'],
                                                                                  ENV_REWARDS['large_lake']['hole_rew'],
                                                                                  ENV_REWARDS['large_lake']['goal_rew']),
            'name': 'large_frozen_lake',
            'readable_name': 'Frozen Lake (15x15)',
        },
        {
            'env': environments.get_windy_cliff_walking_4x12_environment(ENV_REWARDS['cliff_walking']['wind_prob'],
                                                                         ENV_REWARDS['cliff_walking']['step_rew'],
                                                                         ENV_REWARDS['cliff_walking']['fall_rew'],
                                                                         ENV_REWARDS['cliff_walking']['goal_rew']),
            'name': 'cliff_walking',
            'readable_name': 'Cliff Walking (4x12)',
        },
    ]

    # Set up experiments
    experiment_details = []
    for env in envs:
        env['env'].seed(seed)
        logger.info('{}: State space: {}, Action space: {}'.format(env['readable_name'], env['env'].unwrapped.nS,
                                                                   env['env'].unwrapped.nA))
        experiment_details.append(experiments.ExperimentDetails(
            env['env'], env['name'], env['readable_name'],
            threads=threads,
            seed=seed
        ))

    if verbose:
        logger.info("----------")
    print('\n\n')
    logger.info("Running experiments")

    timings = {} # Dict used to report experiment times (in seconds) at the end of the run

    # Run Policy Iteration (PI) experiment
    if args.policy or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.PolicyIterationExperiment, 'PI', verbose, timings, \
                       MAX_STEPS['pi'], NUM_TRIALS['pi'], theta=PI_THETA, discounts=PI_DISCOUNTS)

    # Run Value Iteration (VI) experiment
    if args.value or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.ValueIterationExperiment, 'VI', verbose, timings, \
                       MAX_STEPS['vi'], NUM_TRIALS['vi'], theta=VI_THETA, discounts=VI_DISCOUNTS)

    # Run Q-Learning (QL) experiment
    if args.ql or args.all:
        print('\n\n')
        run_experiment(experiment_details, experiments.QLearnerExperiment, 'QL', verbose, timings, MAX_STEPS['ql'], \
                       NUM_TRIALS['ql'], max_episodes=QL_MAX_EPISODES, max_episode_steps=QL_MAX_EPISODE_STEPS, \
                       min_episodes = QL_MIN_EPISODES, min_sub_thetas=QL_MIN_SUB_THETAS, theta=QL_THETA, \
                       discounts=QL_DISCOUNTS, alphas=QL_ALPHAS, q_inits=QL_Q_INITS, epsilons=QL_EPSILONS, \
                       epsilon_decays=QL_EPSILON_DECAYS)

    # Generate plots
    if args.plot:
        print('\n\n')
        if verbose:
            logger.info("----------")
        logger.info("Plotting results")
        plotting.plot_results(envs)

    # Output timing information
    print('\n\n')
    logger.info(timings)
    print('\n\n')

