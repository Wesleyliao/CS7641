"""
Instructions:
- set BASE_DIR at the top
- Place me in the `assignment3` directory of chadcode
- Change the ds#_names and component/cluster counts in __main__

Output:
    6 files of summarized data

author: Geoff Von Allen
"""

from collections import namedtuple

import pandas as pd

BASE_DIR = 'output'
AlgoDS = namedtuple('AlgoDS', 'folder param_name dim_val')
benchmark = AlgoDS('benchmark', '', '')


def get_non_cluster_results(algo, ds_name, bench=None):
    suffix = 'dim_red.csv'

    # Assignment 1 Architecture
    ass1 = pd.read_csv(f'{BASE_DIR}/{algo.folder}/{ds_name}_ass1_{suffix}').T
    ass1 = ass1.rename(columns={0: f'ass1_{algo.folder}_dr'})


    # Sort chadcode NN grid search by rank
    dr_results = pd.read_csv(f'{BASE_DIR}/{algo.folder}/{ds_name}_{suffix}')
    sorted_ = dr_results.sort_values(by=['rank_test_score'])

    # Best overall - any dim reduction or architecture
    overall = sorted_.T.iloc[:,0]
    overall.name = f'{algo.folder}_overall_best'

    # Get best overall using current algo n_components param
    overall_n = sorted_[sorted_[algo.param_name] == algo.dim_val]
    if overall_n.empty:
        # Dim value was never tested
        print(f"\tDim param {algo.param_name} with value {algo.dim_val}"\
        f" not tested for {algo.folder}")
        return ass1

    overall_n = overall_n.T.iloc[:,0]
    overall_n.name = f'{algo.folder}_overall_best_n={algo.dim_val}'

    # Get best using benchmark chadcode architecture
    alpha_param = 'param_NN__alpha'
    nn_arch = 'param_NN__hidden_layer_sizes'

    # filter for algo specific dim and benchmark nn params
    # Already sorted so best in first position
    filter_  = \
        (sorted_[algo.param_name] == algo.dim_val) & \
        (sorted_[alpha_param] == bench[alpha_param]) & \
        (sorted_[nn_arch] == bench[nn_arch])
    best_bench = sorted_[filter_].T.iloc[:,0]
    best_bench.name = f'{algo.folder}_best_benchmark'
    final = pd.concat([ass1, best_bench, overall_n, overall], axis=1)
    return final


def get_benchmark_results(algo, ds_name):
    # Assignment 1 Architecture
    r = pd.read_csv(f'{BASE_DIR}/{algo.folder}/{ds_name}_ass1_nn_bmk.csv').T
    r = r.rename(columns={0: f'ass1_benchmark'})

    # Best 'New' Architecture on non reduced dataset
    new_best = pd.read_csv(f'{BASE_DIR}/{algo.folder}/{ds_name}_nn_bmk.csv')
    new_best = new_best.sort_values(by=['rank_test_score'])
    new_best = new_best.T.iloc[:,0]
    new_best.name = 'new_benchmark_no_dr'
    final = pd.concat([r, new_best], axis=1)
    return final


def get_cluster_results(algo, cluster_type, ds_name, bench=None):
    # Sort chadcode NN grid search by rank
    c_results = pd.read_csv(f'{BASE_DIR}/{algo.folder}/clustering/{ds_name}_cluster_{cluster_type}.csv')
    sorted_ = c_results.sort_values(by=['rank_test_score'])
    
    # Get best overall using any num clusters or architecrures
    overall = sorted_.T.iloc[:,0]
    overall.name = f'{algo.folder}_overall_best'

    # Get best overall with selected number of clusters among all nn arch
    overall_k = sorted_[sorted_[algo.param_name] == algo.dim_val]
    if overall_k.empty:
        # Dim value was never tested
        print(f"\tDim param {algo.param_name} with value {algo.dim_val}"\
        f" not tested for {algo.folder}")
        return overall

    overall_k = overall_k.T.iloc[:,0]
    overall_k.name = f'{algo.folder}_overall_best_k={algo.dim_val}'

    # Get best using benchmark chadcode architecture
    alpha_param = 'param_NN__alpha'
    nn_arch = 'param_NN__hidden_layer_sizes'

    # filter for algo specific dim and benchmark nn params
    # Already sorted so best in first position
    filter_  = \
        (sorted_[algo.param_name] == algo.dim_val) & \
        (sorted_[alpha_param] == bench[alpha_param]) & \
        (sorted_[nn_arch] == bench[nn_arch])
    best_bench = sorted_[filter_].T.iloc[:,0]
    best_bench.name = f'{algo.folder}_best_benchmark'
    final = pd.concat([best_bench, overall_k, overall], axis=1)
    return final


def run_non_clustered_nn(ds1, ds2):
    print("{:-^50}".format(" Running Non Clustering "))
    dss = [ds1, ds2] 
    for ds in dss:
        print(f"Running Dataset: {ds['name']}")
        result = get_benchmark_results(benchmark, ds['name'])
        bench = result['new_benchmark_no_dr'].T
        for algo in ds['algos']:
            print(f"\tFetching: {algo.folder}")
            r = get_non_cluster_results(algo, ds['name'], bench)
            result = pd.concat([result, r], axis=1)
        result.to_csv(f'non_cluster_results_{ds["name"]}.csv')


def run_clustered_nn(*args):
    print("{:-^50}".format(" Running Clustering "))
    gmms = [x for x in args if x['cluster'] == 'GMM']
    kms = [x for x in args if x['cluster'] == 'kmeans']

    for ds in gmms:
        print(f"Running Dataset: {ds['name']} for cluster {ds['cluster']}")
        result = get_benchmark_results(benchmark, ds['name'])
        bench = result['new_benchmark_no_dr'].T
        result = result['new_benchmark_no_dr']
        for algo in ds['algos']:
            print(f"\tFetching: {algo.folder} for {ds['cluster']}")
            r = get_cluster_results(algo, ds['cluster'], ds['name'], bench)
            result = pd.concat([result, r], axis=1)
        result.to_csv(f'cluster_results_{ds["name"]}_{ds["cluster"]}.csv')

    for ds in kms:
        print(f"Running Dataset: {ds['name']} for cluster {ds['cluster']}")
        result = get_benchmark_results(benchmark, ds['name'])
        bench = result['new_benchmark_no_dr'].T
        result = result['new_benchmark_no_dr']
        for algo in ds['algos']:
            print(f"\tFetching: {algo.folder} for {ds['cluster']}")
            r = get_cluster_results(algo, ds['cluster'], ds['name'], bench)
            result = pd.concat([result, r], axis=1)
        result.to_csv(f'cluster_results_{ds["name"]}_{ds["cluster"]}.csv')



if __name__ == '__main__':
    # Non Clustering
    # NOTE: Change to your ds names that are used in filenames
    ds1_name = 'fashion'
    ds2_name = 'nba_data'

    # NOTE: Change last field to your components for each DR
    ds1_nc = {
        'name': ds1_name,
        'algos': [
            AlgoDS('ICA', 'param_ica__n_components', 0),
            AlgoDS('PCA', 'param_pca__n_components', 0),
            AlgoDS('RP', 'param_rp__n_components', 0),
            AlgoDS('RF', 'param_filter__n', 0),
            ]
        }
    ds2_nc = {
        'name': ds2_name, 
        'algos': [
            AlgoDS('ICA', 'param_ica__n_components', 0),
            AlgoDS('PCA', 'param_pca__n_components', 0),
            AlgoDS('RP', 'param_rp__n_components', 0),
            AlgoDS('RF', 'param_filter__n', 0),
            ]
        }
    run_non_clustered_nn(ds1_nc, ds2_nc)

    # Clustering
    param_gmm = 'param_gmm__n_components'
    param_km = 'param_km__n_clusters'
    gmm = 'GMM'
    km = 'kmeans'

    # NOTE: Change last field to your selected cluster count
    ds1_gmm = {
        'name': ds1_name,
        'cluster': gmm,
        'algos': [
            AlgoDS('benchmark', param_gmm, 0),
            AlgoDS('ICA', param_gmm, 0),
            AlgoDS('PCA', param_gmm, 0),
            AlgoDS('RP', param_gmm, 0),
            AlgoDS('RF', param_gmm, 0),
            ]
        }
    ds2_gmm = {
        'name': ds2_name, 
        'cluster': gmm,
        'algos': [
            AlgoDS('benchmark', param_gmm, 0),
            AlgoDS('ICA', param_gmm, 0),
            AlgoDS('PCA', param_gmm, 0),
            AlgoDS('RP', param_gmm, 0),
            AlgoDS('RF', param_gmm, 0),
            ]
        }
    ds1_km = {
        'name': ds1_name,
        'cluster': km,
        'algos': [
            AlgoDS('benchmark', param_km, 0),
            AlgoDS('ICA', param_km, 0),
            AlgoDS('PCA', param_km, 0),
            AlgoDS('RP', param_km, 0),
            AlgoDS('RF', param_km, 0),
            ]
        }
    ds2_km = {
        'name': ds2_name, 
        'cluster': km,
        'algos': [
            AlgoDS('benchmark', param_km, 0),
            AlgoDS('ICA', param_km, 0),
            AlgoDS('PCA', param_km, 0),
            AlgoDS('RP', param_km, 0),
            AlgoDS('RF', param_km, 0),
            ]
        }

    run_clustered_nn(ds1_gmm, ds2_gmm, ds1_km, ds2_km)



