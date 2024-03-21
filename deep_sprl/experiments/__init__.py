# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .point_mass_2d_experiment import PointMass2DExperiment
from .maze_experiment import MazeExperiment
from .minigrid_a_experiment import MinigridAExperiment
from .minigrid_b_experiment import MinigridBExperiment
from .minigrid_c_experiment import MinigridCExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'PointMass2DExperiment', 'MazeExperiment', 'MinigridAExperiment',
           'MinigridBExperiment', 'MinigridCExperiment', 'Learner']
