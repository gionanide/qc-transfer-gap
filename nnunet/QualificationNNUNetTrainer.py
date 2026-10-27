"""The official nnU-Net optimization core, with the checkpoint trainer name retained.

The original qualification mixin only instrumented finiteness, step counts,
timings and audit files. It did not replace official optimization steps.
This portable class omits those machine-specific audit-output hooks.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class QualificationNNUNetTrainer(nnUNetTrainer):
    pass
