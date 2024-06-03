from __future__ import annotations  # to enable: @classmethod; def func(cls) -> CLASS_NAME
import inspect
import numpy as np
import copy
import matplotlib.pyplot as plt
from typing import Union, Callable, List, Tuple, Dict, Optional, Literal
from scipy.interpolate import RegularGridInterpolator, interp1d


class ErrorModel:
    """This class implements an error model which gives the margin for a certain confidence interval"""

    DimRedMethod = Literal["worst case", "average"]

    def __init__(
        self,
        mean_interpolator: Union[interp1d, RegularGridInterpolator],
        std_interpolator: Union[interp1d, RegularGridInterpolator],
        ndim: int,
        lognormal: bool,
        bounds_error: bool = True,
        model_info: Optional[Dict] = None,
    ) -> None:
        self.mean_interpolator = mean_interpolator
        self.std_interpolator = std_interpolator
        self.ndim = ndim
        self.lognormal = lognormal
        self._bounds_error = bounds_error
        self.model_info = {} if model_info is None else model_info
        self.model_info["ndim"] = self.ndim

        self.mean_interpolator.bounds_error = self._bounds_error
        self.std_interpolator.bounds_error = self._bounds_error

    def __repr__(self):
        string = str(type(self)) + "\n"
        for k, v in self.model_info.items():
            string += str(k) + " " * max(25 - len(str(k)), 0) + ": " + str(v) + "\n"
        for idx, bounds in enumerate(self.get_boundaries()):
            k = f"x{idx} boundaries"
            string += str(k) + " " * max(25 - len(str(k)), 0) + ": " + str(bounds) + "\n"

        return string

    def __call__(self, xs: np.ndarray, stds_margin: float) -> np.ndarray:
        """Make a prediction"""
        self.assert_inputs(xs)
        if not self.bounds_error:
            xs = self.clip_inputs(xs)

        mean, std = self.mean_interpolator(xs), self.std_interpolator(xs)
        return mean + stds_margin * std if not self.lognormal else np.exp(mean + stds_margin * std)

    def get_mean_std(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the mean and standard deviation"""
        self.assert_inputs(xs)
        if not self.bounds_error:
            xs = self.clip_inputs(xs)
        return self.mean_interpolator(xs), self.std_interpolator(xs)
    
    @property
    def bounds_error(self):
        return self._bounds_error

    @bounds_error.setter
    def bounds_error(self, value: bool):
        self.mean_interpolator.bounds_error = value
        self.std_interpolator.bounds_error = value
        self._bounds_error = value

    def assert_inputs(self, xs: np.ndarray) -> None:
        assert xs.ndim == 1 or (xs.ndim == 2 and xs.shape[1] == self.ndim), f"shape should be [N, {self.ndim}]" + (
            f" or [N]" if self.ndim == 1 else ""
        )

    def clip_inputs(self, xs: np.ndarray) -> np.ndarray:
        boundaries = np.array(self.get_boundaries()).T
        return np.clip(xs, *boundaries)

    def get_boundaries(self) -> List[Tuple[float, float]]:
        """Returns a list with tuples with the boundary values for each dimension: [(dim1_min, dim1_max), ...]"""
        if isinstance(self.mean_interpolator, interp1d):
            return [(self.mean_interpolator.x.min(), self.mean_interpolator.x.max())]
        else:
            return [(g.min(), g.max()) for g in self.mean_interpolator.grid]

    def get_model_with_lower_dimension(
        self, mean_method: DimRedMethod, std_method: DimRedMethod, dim_to_rm: int
    ) -> ErrorModel:
        assert self.ndim == 2, "Error model should be 2D"

        dim_to_keep = 1 - dim_to_rm
        x_mean = self.std_interpolator.grid[dim_to_keep]
        if mean_method == "worst case":
            mean_reduced = self.mean_interpolator.values.max(axis=dim_to_rm)
        elif mean_method == "average":
            mean_reduced = self.mean_interpolator.values.mean(axis=dim_to_rm)
        else:
            raise KeyError
        mean_interpolator = interp1d(x_mean, mean_reduced)

        x_std = self.mean_interpolator.grid[dim_to_keep]
        if std_method == "worst case":
            std_reduced = self.std_interpolator.values.max(axis=dim_to_rm)
        elif std_method == "average":
            std_reduced = self.std_interpolator.values.mean(axis=dim_to_rm)
        else:
            raise KeyError
        std_interpolator = interp1d(x_std, std_reduced)

        return ErrorModel(
            mean_interpolator=mean_interpolator,
            std_interpolator=std_interpolator,
            ndim=1,
            lognormal=self.lognormal,
            bounds_error=self.bounds_error,
            model_info={
                "update": f"Dimension {dim_to_rm} of this model has been removed.",
                "mean reduction meth.": mean_method,
                "std reduction meth.": std_method,
                **self.model_info,
            },
        )

    def plot(self) -> None:
        if self.ndim == 1 and isinstance(self.mean_interpolator, interp1d):
            plot_error_model_1D(self)
        elif self.ndim == 2 and isinstance(self.mean_interpolator, RegularGridInterpolator):
            plot_error_model_2D(self)
        else:
            raise TypeError


class LongErrorRateScaleFunction:
    """Make a scale function object to be able to copy it and serialize it."""

    def __init__(self, long_dt_std_scale_function):
        self.scale_error_model = copy.deepcopy(long_dt_std_scale_function)

    def __call__(self, curvature: float) -> float:
        curvature = self.scale_error_model.clip_inputs(curvature)
        std0 = self.scale_error_model.std_interpolator(np.array([0]))  # std at zero
        std1 = self.scale_error_model.std_interpolator(np.array([curvature]))  # std at the current curvature
        return float(std1 / std0)


class ErrorModelWithStdScaleFunc(ErrorModel):
    """This subclass adds the possibility to give a function to scale the std"""

    def __init__(
        self,
        mean_interpolator: Union[interp1d, RegularGridInterpolator],
        std_interpolator: Union[interp1d, RegularGridInterpolator],
        std_scale_func: Callable[[], float],
        ndim: int,
        lognormal: bool,
        bounds_error: bool = True,
        model_info: Optional[Dict] = None,
    ) -> None:
        super().__init__(mean_interpolator, std_interpolator, ndim, lognormal, bounds_error, model_info)
        self.std_scale_func = copy.deepcopy(std_scale_func)
        self.model_info["std_scale_func signature"] = inspect.signature(self.std_scale_func)

    @classmethod
    def from_error_model(
        cls,
        base_model: ErrorModel,
        std_scale_func: Callable[[], float],
    ) -> ErrorModelWithStdScaleFunc:
        """Add an std scale function to an existing error model."""
        return cls(
            mean_interpolator=base_model.mean_interpolator,
            std_interpolator=base_model.std_interpolator,
            std_scale_func=std_scale_func,
            ndim=base_model.ndim,
            lognormal=base_model.lognormal,
            bounds_error=base_model.bounds_error,
            model_info=base_model.model_info,
        )

    def __call__(self, xs: np.ndarray, stds_margin: float, **kwargs) -> np.ndarray:
        """Make a prediction: additional parameters are passed to the std_scale_function"""
        self.assert_inputs(xs)
        if not self.bounds_error:
            xs = self.clip_inputs(xs)
        mean, std = self.mean_interpolator(xs), self.std_interpolator(xs)
        std *= self.std_scale_func(**kwargs)
        return mean + stds_margin * std if not self.lognormal else np.exp(mean + stds_margin * std)

    def get_mean_std(self, xs: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the mean and standard deviation"""
        self.assert_inputs(xs)
        if not self.bounds_error:
            xs = self.clip_inputs(xs)
        mean, std = self.mean_interpolator(xs), self.std_interpolator(xs)
        std *= self.std_scale_func(**kwargs)
        return mean, std

def plot_error_model_2D(error_model) -> None:
    fig, axs = plt.subplots(1, 2, layout="constrained", figsize=(9, 3))

    ax = axs[0]
    x_grid = np.stack(np.meshgrid(*error_model.mean_interpolator.grid), axis=2)
    X, Y = x_grid[..., 0], x_grid[..., 1]
    Z_mean = error_model(x_grid.reshape(-1, 2), stds_margin=0.0).reshape(x_grid.shape[:2])

    CS = ax.contourf(X, Y, Z_mean, 10, cmap="viridis")
    ax.set_title("mean")
    ax.set_xlabel(str(error_model.model_info.get("param1", None)))
    ax.set_ylabel(str(error_model.model_info.get("param2", None)))
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS)
    # cbar.ax.set_ylabel('error mean')

    ax = axs[1]
    Z_std = error_model(x_grid.reshape(-1, 2), stds_margin=1.0).reshape(x_grid.shape[:2]) - Z_mean
    CS = ax.contourf(X, Y, Z_std, 10, cmap="viridis")
    ax.set_title("standard deviation")
    ax.set_xlabel(str(error_model.model_info.get("param1", None)))
    ax.set_ylabel(str(error_model.model_info.get("param2", None)))

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS)
    # cbar.ax.set_ylabel('error std')
    fig.suptitle(str(error_model.model_info.get("error", None)) + " model")

    plt.show()


def plot_error_model_1D(error_model) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    x = error_model.mean_interpolator.x
    y = error_model(x, stds_margin=0.0)
    ax.plot(x, y, label="mean", color="tab:red")
    for k in (1, 2):
        ub = error_model(x, stds_margin=k)
        lb = error_model(x, stds_margin=-k)
        ax.fill_between(x, lb, ub, color="tab:blue", alpha=0.4 - 0.05 * k, label=rf"${k}\sigma$")
    ax.legend()
    ax.set_title(str(error_model.model_info.get("error", None)) + " model")
    ax.set_xlabel(str(error_model.model_info.get("param1", None)))
    ax.set_ylabel("error " + str(error_model.model_info.get("unit", "")))
    plt.show()
