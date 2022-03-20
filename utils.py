from typing import List, Tuple, Union, Callable

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import numpy as np
from sklearn.metrics import precision_score


def initialize_random_generator(
    random_seed: Union[int, RandomState] = None
) -> RandomState:
    if random_seed is None:
        return RandomState(MT19937(SeedSequence()))
    elif isinstance(random_seed, RandomState):
        return random_seed
    else:
        return RandomState(MT19937(SeedSequence(random_seed)))


def generate_random_labels(
    n_samples: int,
    class_proportions: List[int],
    random_seed: Union[int, RandomState] = None,
) -> np.ndarray:

    assert np.sum(class_proportions) == 1

    rs = initialize_random_generator(random_seed)

    labels = rs.rand(n_samples)
    n_classes = len(class_proportions)
    cum_props = np.cumsum([0, *class_proportions])
    for idx in range(0, n_classes):
        labels[
            np.logical_and(labels >= cum_props[idx], labels < cum_props[idx + 1])
        ] = idx

    return labels


def generate_random_predictions(
    y: np.ndarray,
    class_proportions: List[int],
    random_seed: Union[int, RandomState] = None,
) -> np.ndarray:

    n_samples = len(y)
    return generate_random_labels(n_samples, class_proportions, random_seed)


class RandomClassificationExperiment:
    def __init__(
        self,
        p_vec: np.ndarray,
        q_vec: np.ndarray = None,
        metric_func: Callable = precision_score,
        metric_name: str = "precision",
        param_name: str = "q",
        p_vec_highres: np.ndarray = np.arange(0.01, 1, 0.01),
        q_vec_highres: np.ndarray = np.arange(0.01, 1, 0.01),
        aspect: str = "equal",
        xlimits: Tuple = (0, 1),
        ylimits: Tuple = (0, 1),
        figsize: Tuple = (5, 5),
        animate: bool = False,
        interval: int = 500,
        random_seed: int = 42,
        n_samples: int = 1000,
        n_iter: int = 10,
        q_equals_p: bool = False,
    ):
        self.p_vec = p_vec
        self.q_vec = q_vec
        self.metric_func = metric_func
        self.metric_name = metric_name
        self.param_name = param_name
        self.p_vec_highres = p_vec_highres
        self.q_vec_highres = q_vec_highres
        self.aspect = aspect
        self.xlimits = xlimits
        self.ylimits = ylimits
        self.figsize = figsize
        self.animate = animate
        self.interval = interval
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.n_iter = n_iter
        self.q_equals_p = q_equals_p

        assert self.param_name == "q" or self.param_name == "p"

        # If q_vec is not specified, set q = p
        if self.q_vec is None:
            self.q_equals_p = True
            self.param_name = "q = p"
            self.animate = False

        if self.param_name in ["q", "q = p"]:
            self.x_vec = self.p_vec
            self.x_vec_highres = self.p_vec_highres
            self.x_name = "p"
            self.q_vec_highres = self.q_vec
            self.param_vec = self.q_vec
        elif self.param_name == "p":
            self.x_vec = self.q_vec
            self.x_vec_highres = self.q_vec_highres
            self.x_name = "q"
            self.p_vec_highres = self.p_vec
            self.param_vec = self.p_vec

        self.rs = initialize_random_generator(self.random_seed)

    def run(self):
        self.calculate_random_classification_scores()
        self.calculate_analytic_scores()
        self.plot_results()

    def plot_results(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)
        self.ax.set_xlim(self.xlimits)
        self.ax.set_ylim(self.ylimits)
        self.ax.set_xlabel(self.x_name)
        self.ax.set_ylabel(self.metric_name)
        self.ax.set_aspect(self.aspect)

        if self.animate:
            self.animation = FuncAnimation(
                self.fig,
                self._plot_scores,
                frames=len(self.param_vec),
                interval=self.interval,
            )
            self.paused = False
            self.fig.canvas.mpl_connect("button_press_event", self._toggle_pause)
        else:
            if self.param_name == "q = p":
                self._plot_scores(0)
                self.ax.set_title(self.param_name)
            else:
                for idx, param in enumerate(self.param_vec):
                    legend_label = f"{self.param_name} = {param:.1f}"
                    self._plot_scores(idx, legend_label)
                self.ax.legend(loc="best")

    def _plot_scores(self, idx, legend_label=None):
        if self.animate:
            for artist in self.ax.lines + self.ax.collections:
                artist.remove()
                self.ax.set_title(f"{self.param_name} = {self.param_vec[idx]:.1f}")
        self.ax.plot(
            self.x_vec_highres,
            self.analytic_scores[:, idx],
            label=legend_label,
            zorder=1,
        )
        self.ax.scatter(self.x_vec, self.med_scores[:, idx], color="k", zorder=2)
        self.ax.errorbar(
            self.x_vec,
            self.med_scores[:, idx],
            self.iqr_scores[:, idx],
            ls="none",
            color="k",
            zorder=2,
        )

    def calculate_random_classification_scores(self):
        p_vec_len = len(self.p_vec)
        q_vec_len = 1 if self.q_equals_p else len(self.q_vec)
        self.scores = np.zeros((self.n_iter, p_vec_len, q_vec_len))
        for p_idx, p in enumerate(self.p_vec):
            q_vec = [p] if self.q_equals_p else self.q_vec
            for q_idx, q in enumerate(q_vec):
                for iter_idx in range(self.n_iter):
                    y = generate_random_labels(
                        self.n_samples, [1 - p, p], random_seed=self.rs
                    )
                    y_hat = generate_random_predictions(
                        y, [1 - q, q], random_seed=self.rs
                    )
                    self.scores[iter_idx, p_idx, q_idx] = self.metric_func(y, y_hat)

        self.med_scores = np.median(self.scores, 0)
        self.iqr_scores = np.subtract(*np.percentile(self.scores, [75, 25], 0))

        if self.param_name == "p":
            self.med_scores = self.med_scores.T
            self.iqr_scores = self.iqr_scores.T

        return self

    def calculate_analytic_scores(self):
        p = self.p_vec_highres[:, np.newaxis]
        q = self.p_vec_highres if self.q_equals_p else self.q_vec_highres[:, np.newaxis]

        if self.metric_name == "precision":
            if self.q_equals_p:
                self.analytic_scores = p * np.ones(q.shape)
            else:
                self.analytic_scores = p.dot(np.ones(q.T.shape))

        if self.param_name == "p":
            self.analytic_scores = self.analytic_scores.T

        return self

    def _toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused


if __name__ == "__main__":
    anim = RandomClassificationExperiment(
        p_vec=np.arange(0.1, 1, 0.1),
        q_vec=None,
        metric_func=precision_score,
        metric_name="precision",
        param_name="p",
        animate=False,
    )
