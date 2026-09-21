# pytest loads this before collecting test modules. Importing torch here, ahead of the
# numpy-only data tests, avoids the Anaconda crash where two OpenMP runtimes load in the
# wrong order (see utils/plotting.py). Harmless on pip installs, where there is no clash.
import torch  # noqa: F401
