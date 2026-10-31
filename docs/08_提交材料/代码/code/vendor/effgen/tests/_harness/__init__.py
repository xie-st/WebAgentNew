"""Support code for running the test suite itself.

The modules here are harness, not subject: they describe lanes, time them, run the
suite without the ambient state of the machine it runs on, and read the register of
tests known to fail for a reason outside the tree.
"""
