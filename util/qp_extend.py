from questplus.qp import QuestPlus
from typing import Optional, Sequence
import numpy as np


class QuestPlusNorm(QuestPlus):
    def __init__(self, *,
                 intensities: Sequence,
                 means: Sequence,
                 sds: Sequence,
                 lower_asymptotes: Sequence,
                 lapse_rates: Sequence,
                 prior: Optional[dict] = None,
                 responses: Sequence = ('Yes', 'No'),
                 stim_scale: str = 'linear',
                 stim_selection_method: str = 'min_entropy',
                 stim_selection_options: Optional[dict] = None,
                 param_estimation_method: str = 'mean'):
        super().__init__(stim_domain=dict(intensity=intensities),
                         param_domain=dict(mean=means,
                                           sd=sds,
                                           lower_asymptote=lower_asymptotes,
                                           lapse_rate=lapse_rates),
                         outcome_domain=dict(response=responses),
                         prior=prior,
                         stim_scale=stim_scale,
                         stim_selection_method=stim_selection_method,
                         stim_selection_options=stim_selection_options,
                         param_estimation_method=param_estimation_method,
                         func='norm_cdf')

    @property
    def intensities(self) -> np.ndarray:
        """
        Stimulus intensity or contrast domain.
        """
        return self.stim_domain['intensity']

    @property
    def means(self) -> np.ndarray:
        """
        The mean parameter domain.
        """
        return self.param_domain['mean']

    @property
    def sds(self) -> np.ndarray:
        """
        The standard deviation parameter domain.
        """
        return self.param_domain['sd']

    @property
    def lower_asymptotes(self) -> np.ndarray:
        """
        The lower asymptote parameter domain.
        """
        return self.param_domain['lower_asymptote']

    @property
    def lapse_rates(self) -> np.ndarray:
        """
        The lapse rate parameter domain.
        """
        return self.param_domain['lapse_rate']

    @property
    def responses(self) -> np.ndarray:
        """
        The response (outcome) domain.
        """
        return self.outcome_domain['response']

    @property
    def next_intensity(self) -> float:
        """
        The intensity or contrast to present next.
        """
        return super().next_stim['intensity']

    def update(self, *,
               intensity: float,
               response: str) -> None:
        """
        Inform QUEST+ about a newly gathered measurement outcome for a given
        stimulus intensity or contrast, and update the posterior accordingly.

        Parameters
        ----------
        intensity
            The intensity or contrast of the presented stimulus.

        response
            The observed response.

        """
        super().update(stim=dict(intensity=intensity),
                       outcome=dict(response=response))


