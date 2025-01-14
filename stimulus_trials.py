# %% a file that defines the stimulus file names

import numpy as np


class ContrastStimulus:
    """
    A class that defines the stimulus conditions for contrast experiments.
    Use this class to iterate over all the conditions.
    Consistent usage across scripts is recommended.

    Usage:
    stim = ContrastStimulus()
    for angle, contrast, trial in stim:
        # do something with the angle, contrast
    """

    def __init__(self):
        self.angles = np.linspace(0, 315, 8)
        self.contrasts = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
        self.trials = range(10)
        self.blanks = 30

    def get_all_result_paths(self, basedir, network_option):
        path_list = []
        for angle, contrast, trial in self:
            path_list.append(
                f"{basedir}/contrasts_{network_option}/angle{int(angle)}_contrast{contrast}_trial{trial}"
            )
        return path_list

    def get_shape(self):
        return (self.blanks, (len(self.angles), len(self.contrasts), len(self.trials)))

    # make an iterator for each conditions
    def __iter__(self):
        self.angle_idx = 0
        self.contrast_idx = 0
        self.trial_idx = 0
        self.blank_idx = 0
        self.blank_done = False
        return self

    def __next__(self):
        # first, do iterattions on blank trials. Angle is fixed to 0.
        if not self.blank_done:
            angle = 0
            contrast = 0.0
            trial = self.blank_idx
            self.blank_idx += 1
            if self.blank_idx >= self.blanks:
                self.blank_done = True
            return angle, contrast, trial

        # When done, do the actual stimuli.
        if self.angle_idx >= len(self.angles):
            raise StopIteration
        angle = self.angles[self.angle_idx]
        contrast = self.contrasts[self.contrast_idx]
        trial = self.trials[self.trial_idx]

        self.trial_idx += 1
        if self.trial_idx >= len(self.trials):
            self.trial_idx = 0
            self.contrast_idx += 1
            if self.contrast_idx >= len(self.contrasts):
                self.contrast_idx = 0
                self.angle_idx += 1

        return angle, contrast, trial
