"""Offline training for the perception verifiers (workstream C). Runs on Colab or a laptop, never on the drone.

PRIMARY: train_cnn.py (small CNN -> ONNX).  [STRETCH]: train_gbm.py (gradient boosting on blob features -> ONNX).
Shared preprocessing comes from perception/ (patches.py, features.py) so training and inference cannot drift.
"""
