import argparse
import logging
import sys

import random as rand
import numpy as np

import experiments
from experiments import plotting
from datetime import datetime
from data import loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, dim, skiprerun, verbose, timings):
    t = datetime.now()
    for details in experiment_details:
        exp = experiment(details, verbose=verbose)

        if not skiprerun:
            logger.info("Running {} experiment: {} ({})".format(timing_key, details.ds_readable_name, dim))
            exp.perform()

        if dim is not None:
            logger.info("Running with dimension {}".format(dim))
            exp.perform_cluster(dim)
    t_d = datetime.now() - t
    timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform some UL and DR')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--dim', type=int, help='The dim parameter to use for clustering with a specific experiment '
                                                '(This MUST be used with a specific experiment)')
    parser.add_argument('--skiprerun', action='store_true',
                        help='If true, do not re-run the main experiment before clustering '
                             '(This MUST be used with --dim and a specific experiment)')
    parser.add_argument('--nba', action='store_true', help='Run only nba')
    parser.add_argument('--fashion', action='store_true', help='Run only fashion')
    parser.add_argument('--benchmark', action='store_true', help='Run the benchmark experiments')
    parser.add_argument('--ica', action='store_true', help='Run the ICA experiments')
    parser.add_argument('--pca', action='store_true', help='Run the PCA experiments')
    parser.add_argument('--lda', action='store_true', help='Run the LDA experiments')
    parser.add_argument('--svd', action='store_true', help='Run the SVD experiments')
    parser.add_argument('--rf', action='store_true', help='Run the RF experiments')
    parser.add_argument('--rp', action='store_true', help='Run the RP experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--plot', action='store_true', help='Plot data results')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    if args.dim or args.skiprerun:
        if not args.ica and not args.pca and not args.rf and not args.rp and not args.lda and not args.svd:
            logger.error("Cannot specify dimension/skiprerun without specifying a specific experiment")
            parser.print_help()
            sys.exit(1)

    if args.skiprerun and not args.dim:
        logger.error("Cannot specify skiprerun without specifying a specific experiment")
        parser.print_help()
        sys.exit(1)

    if args.nba and args.fashion:
        logger.error("Can only specify one of '--nba' or '--fashion', not both")
        parser.print_help()
        sys.exit(1)

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 31) - 1)
        logger.info("Using seed {}".format(seed))
        np.random.seed(seed)
        rand.seed(seed)

    logger.info("Loading data")
    logger.info("----------")

    datasets = []
    nba_details = {
            'data': loader.NBAData(verbose=verbose, seed=seed),
            'name': 'nba_data',
            'readable_name': 'NBA Shots',
        }
    fashion_details = {
            'data': loader.FashionData(verbose=verbose, seed=seed),
            'name': 'fashion',
            'readable_name': 'Fashion Pictures',
        }
    if args.nba:
        datasets.append(nba_details)
    elif args.fashion:
        datasets.append(fashion_details)
    elif not args.nba and not args.fashion:
        datasets.append(nba_details)
        datasets.append(fashion_details)

    experiment_details = []
    for ds in datasets:
        data = ds['data']
        data.load_and_process()
        data.build_train_test_split()
        data.scale_standard()
        experiment_details.append(experiments.ExperimentDetails(
            data, ds['name'], ds['readable_name'],
            threads=threads,
            seed=seed
        ))

    if args.all or args.benchmark or args.ica or args.pca or args.lda or args.svd or args.rf or args.rp:
        if verbose:
            logger.info("----------")

        logger.info("Running experiments")

        timings = {}

        if args.benchmark or args.all:
            run_experiment(experiment_details, experiments.BenchmarkExperiment, 'Benchmark', args.dim, args.skiprerun,
                           verbose, timings)
        if args.ica or args.all:
            run_experiment(experiment_details, experiments.ICAExperiment, 'ICA', args.dim, args.skiprerun,
                           verbose, timings)
        if args.pca or args.all:
            run_experiment(experiment_details, experiments.PCAExperiment, 'PCA', args.dim, args.skiprerun,
                           verbose, timings)
        # if args.lda or args.all:
        #     run_experiment(experiment_details, experiments.LDAExperiment, 'LDA', args.dim, args.skiprerun,
        #                    verbose, timings)
        # if args.svd or args.all:
        #     run_experiment(experiment_details, experiments.SVDExperiment, 'SVD', args.dim, args.skiprerun,
        #                    verbose, timings)
        if args.rf or args.all:
            run_experiment(experiment_details, experiments.RFExperiment, 'RF', args.dim, args.skiprerun,
                           verbose, timings)
        if args.rp or args.all:
            run_experiment(experiment_details, experiments.RPExperiment, 'RP', args.dim, args.skiprerun,
                           verbose, timings)

        logger.info("Timings: {}".format(timings))

    if args.plot:
        if verbose:
            logger.info("----------")

        logger.info("Plotting results")
        plotting.plot_results()
