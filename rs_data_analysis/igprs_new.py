import numpy as np
import pdb
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import gpflow
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import trim_mean
from ConfigSpace import ConfigurationSpace, Float, Categorical
from arlbench.core.algorithms import DQN, PPO, SAC
from autorl_landscape.visualize import (
    LEGEND_FSIZE,
    TITLE_FSIZE,
)

from matplotlib.gridspec import GridSpecFromSubplotSpec 
from itertools import zip_longest
from pandas import DataFrame
from autorl_landscape.analyze.visualization import Visualization

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pandas import DataFrame
from sklearn.base import BaseEstimator


import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator

from autorl_landscape.analyze.visualization import Visualization
from autorl_landscape.run.compare import iqm

import re

# font sizes:
TITLE_FSIZE = 24
LABEL_FSIZE = 24
TICK_FSIZE = 20
LEGEND_FSIZE = 18
# TITLE_FSIZE = 1
# LABEL_FSIZE = 1
# TICK_FSIZE = 1
# LEGEND_FSIZE = 1
# plt.rc("legend", fontsize=LEGEND_FSIZE)
Y_SCALED = 1.0  # make this lower if visualization axis limits are too big in y direction
ZOOM_3D = 0.9

LABELPAD = 10

TICK_POS = np.linspace(0, 1, 4)
TICK_POS_RETURN = np.linspace(0, 1, 6)
DEFAULT_GRID_LENGTH = 51


CMAP = {
    "cmap": sns.color_palette("rocket", as_cmap=True),
    "norm": None,
}
CMAP_DIVERGING = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.5),
}
CMAP_CRASHED = {
    "cmap": LinearSegmentedColormap.from_list("", ["#15161e", "#db4b4b"]),  # Tokyo!
    "norm": TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
}
CMAP_DISCRETIZED = {
    "cmap": sns.color_palette("vlag", as_cmap=True),
    "norm": BoundaryNorm(boundaries=[-0.5, 0.75, 1.25, 4.0], ncolors=255),
}



def get_hp_keys(data, algorithm, environment):
    # Filter the data based on the given algorithm and environment
    filtered_data = data[(data['algorithm'] == algorithm) & (data['env'] == environment)]
    if not filtered_data.empty:
        # Extract the values of top_hp1 and top_hp2
        top_hp1 = filtered_data['top_hp_1'].values[0]
        top_hp2 = filtered_data['top_hp_2'].values[0]
        return top_hp1, top_hp2
    else:
        return []

def get_phase(data, step_count):
    phase_data = data[data["budget"] == step_count].sort_values(by="run_id", ascending=True)
    # phase_data = phase_data[phase_data["seed"] == 2]
    # import pdb; pdb.set_trace()
    best_config = phase_data.loc[phase_data["performance"].idxmax()]
    best_config.drop(["budget", "performance", "seed", "run_id"], inplace=True)
    n_dims = len(best_config)
    phase_data["conf.ls.dims"] = n_dims
    #best_config["conf.l"]
    return phase_data, best_config



def _transpose(ll):
    """Transposes lists of lists (or other things you can iterate over)."""
    return list(map(list, zip_longest(*ll, fillvalue=None)))


def iqm(x, axis: int | None = None):
    """Calculate the interquartile mean (IQM) of x. Return nan where np.mean would also write nan."""
    iqms = trim_mean(x, proportiontocut=0.25, axis=axis)
    means = np.mean(x, axis=axis)
    iqms[np.isnan(means)] = np.nan
    return iqms

def estimate_model_fit(X, y, k: int = 5, metrics: list[callable] | None = None) -> pd.DataFrame:            
    if metrics is None:
        metrics = [mean_squared_error, mean_absolute_error]

    cv = KFold(n_splits=k, shuffle=True, random_state=0)

    data = []
    for i, (train_index, test_index) in enumerate(cv.split(X=X, y=y)):
        X_i = X[train_index]
        Y_i = y[train_index]
        model = gpflow.models.GPR((X_i, Y_i), kernel=gpflow.kernels.SquaredExponential())
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)
        f_mean, _ = model.predict_f(X[test_index])
        y_pred = f_mean.numpy()
        results = {}
        results["fold"] = i
        for metric in metrics:
            results[metric.__name__] = metric(y[test_index], y_pred)
        data.append(results)
    data = pd.DataFrame(data)

    return data


class TripleGPModel(BaseEstimator):
    """Triple GP Model."""

    def __init__(
        self,
        data,
        dtype: type,
        y_col: str = "performance",
        y_bounds: tuple[float, float] | None = None,
        best_conf = None,
        hp_names = None,
        configspace: ConfigurationSpace = None,
        ci: float = 0.95,
    ) -> None:
        super().__init__()#data, dtype, y_col, y_bounds, best_conf, ci)
        gpflow.config.set_default_float(dtype)  # WARNING Global!
        self.data = data
        self.y_col = y_col
        self.y_bounds = y_bounds
        self.ci = ci
        self.best_conf = best_conf
        self.model_layer_names = ["upper", "middle", "lower"]
        self.hp_names = hp_names
        self.dtype = dtype
        self.y_info = "yelp"

        # group runs with the same configuration:
        self.dim_info = hp_names
        conf_groups = data.groupby(["run_id"] + hp_names)
        # all groups (configurations):
        self.x = np.array(list(conf_groups.groups.keys()))[:, 1:]
        """(num_confs, num_ls_dims). LS dimensions are sorted by name"""
        # all evaluations (y values) for a group (configuration):
        #print(conf_groups[y_col].apply(list).values)
        y = np.concatenate(conf_groups[y_col].apply(list))
        y = np.array([np.array(ys) for ys in y])
        #y = y.reshape((-1, len(np.unique(data["seed"]))))
        # handle crashed runs by assigning 0 return:
        self.crashed = np.isnan(y)
        y[np.isnan(y)] = 0

        # scale ls dims into [0, 1] interval:
        for i in range(len(hp_names)):
            if configspace.get_hyperparameter(hp_names[i].split(".")[-1]).__class__ == Float:
                self.x[:, i] = (self.x[:, i] - configspace.get_hyperparameter(hp_names[i]).lower) / (configspace.get_hyperparameter(hp_names[i]).upper - configspace.get_hyperparameter(hp_names[i]).lower)
        # scale y into [0, 1] interval:
        # self.y = (y - y,[0]) / (y_bounds[1] - y_bounds[0])
        self.y = (y - y.min()) / (y.max() - y.min())
        
        """(num_confs, samples_per_conf)"""

        # just all the single evaluation values, not grouped (but still scaled to [0, 1] interval):
        self.x_samples = np.repeat(self.x, self.y.shape[1], axis=0)
        """(num_confs * samples_per_conf, num_ls_dims)"""
        self.y_samples = self.y.reshape(-1, 1)
        """(num_confs * samples_per_conf, 1)"""

        upper_quantile = 1 - ((1 - ci) / 2)
        lower_quantile = 0 + ((1 - ci) / 2)

        # statistical information about each configuration:
        self.y_iqm = iqm(self.y, axis=1).reshape(-1, 1)
        """(num_confs, 1)"""
        # first, select ci quantile:
        self.y_ci_upper = np.quantile(self.y, upper_quantile, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""
        self.y_ci_lower = np.quantile(self.y, lower_quantile, method="median_unbiased", axis=1, keepdims=True)
        """(num_confs, 1)"""

        self._viz_infos: list[Visualization] = [
            Visualization(
                "Raw Return Samples",
                "scatter",
                "graphs",
                self.build_df(self.x_samples, self.y_samples, "performance"),
                {},
                # {"color": "red"},
            )
        ]

    def fit(self):
        """Fit the three GPs to IQM, upper and lower CI."""
        self.iqm_model = gpflow.models.GPR((self.x, self.y_iqm), kernel=gpflow.kernels.SquaredExponential())
        self.upper_model = gpflow.models.GPR((self.x, self.y_ci_upper), kernel=gpflow.kernels.SquaredExponential())
        self.lower_model = gpflow.models.GPR((self.x, self.y_ci_lower), kernel=gpflow.kernels.SquaredExponential())

        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.iqm_model.training_loss, self.iqm_model.trainable_variables)
        opt.minimize(self.upper_model.training_loss, self.upper_model.trainable_variables)
        opt.minimize(self.lower_model.training_loss, self.lower_model.trainable_variables)

    def estimate_iqm_fit(self):
        print("-"*50)
        print("Estimate IQM surface fit")
        data = estimate_model_fit(X=self.x, y=self.y_iqm, k=5)
        for c in data.columns:
            if c is not "fold":
                print(c, data[c].mean(), data[c].std())
        return data
        
    def get_upper(self, x, assimilate_factor: float = 1.0):
        """Return the upper CI estimate of y at the position(s) x."""
        f_mean, _ = self.upper_model.predict_f(x)
        return self._ci_scale(x, f_mean.numpy(), assimilate_factor)

    def get_middle(self, x):
        """Return the IQM estimate of y at the position(s) x."""
        f_mean, _ = self.iqm_model.predict_f(x)
        return f_mean.numpy()

    def get_lower(self, x, assimilate_factor: float = 1.0):
        """Return the lower CI estimate of y at the position(s) x."""
        f_mean, _ = self.lower_model.predict_f(x)
        return self._ci_scale(x, f_mean.numpy(), assimilate_factor)

    @staticmethod
    def get_model_name() -> str:
        """Return name of this model, for naming files and the like."""
        return "igpr_"
    
    def add_viz_info(self, viz_info) -> None:
        """Add a visualization to this model."""
        self._viz_infos.append(viz_info)

    def get_viz_infos(self):
        """Return visualization info(s) for data points used for training the model."""
        return self._viz_infos
    
    def build_df(self, x, y, y_axis_label: str):
        """Helper to construct a `DataFrame` given x points and y readings and a label for y.

        Labels for x are taken from the model.
        """
        assert x.shape[1] == len(self.dim_info)
        return DataFrame(np.concatenate([x, y], axis=1), columns=self.get_ls_dim_names() + [y_axis_label])
    
    def _ci_scale(self, x, y, assimilate_factor: float = 1.0):
        """Assimilate passed y values (assumed to come from `get_upper` or `get_lower`) towards the middle values."""
        if assimilate_factor == 1.0:
            return y

        y_middle = self.get_middle(x)
        return assimilate_factor * y + (1 - assimilate_factor) * y_middle
    
    def get_ls_dim_names(self) -> list[str]:
        """Get the list of hyperparameter landscape dimension names."""
        return self.hp_names
    
    def get_dim_info(self, name: str):
        """Return matching `DimInfo` to a passed name (can be y_info of any dim_info)."""
        return None
    

def create_contour_plot(model, x_dim, y_dim, z_dim, env, algo, grid_length=51, ):
    # Generate a finer grid for contour plot
    x = np.linspace(model.x[:, x_dim].min(), model.x[:, x_dim].max(), grid_length)
    y = np.linspace(model.x[:, y_dim].min(), model.x[:, y_dim].max(), grid_length)

    

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(grid_length):
        for j in range(grid_length):
            point = np.zeros(model.x.shape[1])
            point[x_dim] = X[i, j]
            point[y_dim] = Y[i, j]
            Z[i, j] = model.get_middle(point.reshape(1, -1))

    # Create contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=20, cmap="rocket", fontsize=14)
    plt.colorbar(contour)

    # Mark the peaks and valleys
    peaks = np.where(Z == Z.max())
    valleys = np.where(Z == Z.min())
    plt.scatter(X[peaks], Y[peaks], color='white', marker='^', s=100, edgecolor='black')  # Peaks
    plt.scatter(X[valleys], Y[valleys], color='white', marker='v', s=100, edgecolor='black')  # Valleys

    plt.xlabel(model.hp_names[x_dim].split('.')[-1], fontsize=18)
    plt.ylabel(model.hp_names[y_dim].split('.')[-1], fontsize=18)
    plt.title(f'{z_dim}', fontsize=18)
    plt.show()


def create_3d_surface_plot(model, x_dim, y_dim, z_dim, grid_length=51, env='test', algo='test'):
    # Generate a finer grid for contour plot
    x = np.linspace(model.x[:, x_dim].min(), model.x[:, x_dim].max(), grid_length)
    y = np.linspace(model.x[:, y_dim].min(), model.x[:, y_dim].max(), grid_length)
    X, Y = np.meshgrid(x, y)
    Z_middle = np.zeros_like(X)
    Z_upper = np.zeros_like(X)
    Z_lower = np.zeros_like(X)

    for i in range(grid_length):
        for j in range(grid_length):
            point = np.zeros(model.x.shape[1])
            point[x_dim] = X[i, j]
            point[y_dim] = Y[i, j]
            Z_middle[i, j] = model.get_middle(point.reshape(1, -1))
            Z_upper[i, j] = model.get_upper(point.reshape(1, -1))
            Z_lower[i, j] = model.get_lower(point.reshape(1, -1))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_middle, cmap=sns.color_palette("rocket", as_cmap=True), edgecolor='none', alpha=0.7)
    ax.plot_surface(X, Y, Z_upper, color='lightgray', edgecolor='none', alpha=0.5)
    ax.plot_surface(X, Y, Z_lower, color='lightgray', edgecolor='none', alpha=0.5)
    
    ax.set_xlabel(f'ls.{model.hp_names[x_dim]}')
    ax.set_ylabel(f'ls.{model.hp_names[y_dim]}')
    ax.set_zlabel(z_dim)
    ax.set_title(f'{z_dim} Middle, Upper, and Lower Surfaces')
    plt.show()

def extract_hyperparameters(log_file):
    """Extract hyperparameters from the run_arlbench.log file."""
    hp_config = {}
    with open(log_file, 'r') as file:
        inside_hp_config = False
        for line in file:
            # Detect when the hp_config section starts
            if line.strip() == "hp_config:":
                inside_hp_config = True
                continue
            # Stop reading when another config section starts
            if inside_hp_config and (not line.startswith(' ') or line.strip() == ""):
                break
            # Extract key-value pairs from the hp_config section
            if inside_hp_config:
                match = re.match(r'\s*(\S+):\s+(.+)', line)
                if match:
                    key, value = match.groups()
                    # Try to convert to appropriate data types
                    try:
                        if value.lower() == "true":
                            value = True
                        elif value.lower() == "false":
                            value = False
                        else:
                            value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass
                    # hp_config[key] = value
                    hp_config["hp_config." + key] = value  # Add prefix here
    return hp_config


def parse_folder_structure_with_hyperparams(root_folder):
    data = []

    # Iterate over each seed folder
    for seed_folder in os.listdir(root_folder):
        if seed_folder == '.DS_Store':
            continue
        
        seed_path = os.path.join(root_folder, seed_folder)

        # Check if seed_path is a directory before proceeding
        if not os.path.isdir(seed_path):
                continue  # Skip if it's not a directory

        # Iterate over each run_id folder in the seed folder
        # import pdb; pdb.set_trace()
        for run_id_folder in os.listdir(seed_path):
            
            run_id_path = os.path.join(seed_path, run_id_folder)

            # Read the emissions.csv file for the current run_id
            emissions_file = os.path.join(run_id_path, 'evaluation.csv')
            log_file = os.path.join(run_id_path, 'run_arlbench.log')
            
            if os.path.exists(emissions_file) and os.path.exists(log_file):
                emissions_data = pd.read_csv(emissions_file)
                hp_config = extract_hyperparameters(log_file)


                # import pdb; pdb.set_trace()
                # For each step, collect the relevant information
                for _, row in emissions_data.iterrows():
                    steps = row['steps']
                    returns = row['returns']
                    
                    # Combine emissions data with hyperparameters
                    run_data = {
                        'run_id': int(run_id_folder),
                        'budget': int(steps),
                        'performance': returns,
                        'seed': int(seed_folder),
                    }
                    run_data.update(hp_config)  # Add hyperparameters to the run data
                    
                    data.append(run_data)
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    
    # Parse the folder structure to create the DataFrame
    root_folder = '../results_finished/sobol_landscaping/box2d/sac/sac_BipedalWalker-v3'  # Replace with your actual folder path
    data = parse_folder_structure_with_hyperparams(root_folder)
    # import pdb; pdb.set_trace()
    data.to_csv('./Landscape-Data/' + root_folder.split('/')[-1] + '.csv', index=False)
    
    algorithms = {
        "ppo": PPO,
        "dqn": DQN,
        "sac": SAC,
    }
    
    
    # data = pd.read_csv('./Landscape-Data/dqn_CartPole-v1.csv')
    top_df = pd.read_csv('./top_2_importances.csv')
    
    env= "box2d_bipedal_walker"
    algorithm = 'sac'

    # Load the data fron huggingface and get the last phase
    old_data = pd.read_csv(f"../results_combined/sobol/{env}_{algorithm}.csv")
    
    # print(data.head()) 
    # old_data['performance'] = old_data['performance'] * -1
    # data = old_data
    
    # import pdb; pdb.set_trace() 

    phases = data['budget'].unique()
    
    # import pdb; pdb.set_trace()
    
    configspace = ConfigurationSpace()

    # Get top hyperparameters
    top_hps = get_hp_keys(top_df, algorithm, env)
    top_hps = ["buffer_beta","reward_scale"]
    

    # import pdb; pdb.set_trace()


    # Get hyperparameter configuration space
    default_configspace = algorithms[algorithm].get_hpo_search_space().get_hyperparameter_names()
    print(default_configspace)
    
    # import pdb; pdb.set_trace()
    
    # Handle continuous vs categorical hyperparameters
    for c in set(default_configspace):
        if c in ["buffer_prio_sampling", "use_target_network", "alpha_auto", "normalize_observations"]: 
            hp = Categorical(c, [True, False])
        else:
            key = [a for a in data.keys() if c in a]
            if len(key) > 0:
                    key = key[0]
                    hp_min = data[key].min()
                    hp_max = data[key].max()
                    bounds = (min(0, hp_min-0.1*hp_min), hp_max+0.1*hp_max)
                    hp = Float(c, bounds=bounds)
                    configspace.add_hyperparameter(hp)

    # Process each phase in the dataset
    # for phase_steps in phases:
    phase_steps = phases[-1]
    # import pdb; pdb.set_trace()
    phase_data, best_conf = get_phase(data, phase_steps)
    phase_data['max_return'] = phase_data['performance'].max()
    Y_BOUNDS = (phase_data['max_return'].min(), phase_data['max_return'].max())
    hp_list = ['hp_config.' + hp for hp in top_hps]

    # Create and fit the TripleGPModel
    model = TripleGPModel(phase_data, np.float64, "performance", Y_BOUNDS, None, hp_list, configspace)
    model.fit()

    # import pdb ; pdb.set_trace()

    # Create contour plot
    create_contour_plot(model, x_dim=0, y_dim=1, z_dim='Eval Returns', env=env, algo=algorithm)