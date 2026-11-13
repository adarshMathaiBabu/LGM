"""Implementation of Generative Adversarial Networks.

GANs have an absurd number of variations, specialized models, little tricks to improve training,
etc. This would be impossible to put into a single repository. We provide a basic version with a few
improvements, without which they likely wouldn't cooperate at all.
"""
from .fun import optimization_inference
from .model import Discriminator, GAN, Generator
from .training import GANTrainingModuleParallel, GANTrainingModuleSequential
