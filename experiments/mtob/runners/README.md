# Runners

The reusable runner implementation is the sibling `mtob_grammar` Python
package. The executable entry point is `../run_experiment.py`; generated Slurm
wrappers live under `../jobs`.

This directory is retained as the explicit runner boundary in the isolated
experiment layout without duplicating executable source files.
