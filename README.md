Installation instructions

The scripts themselves are Python 2 and 3 compatible, so should be usable by existing Python 2 installs with all necessary dependencies.

For new installations, it is recommended to use Python 3 (Python 2 is no longer being maintained) and Conda (fastest way to install RDkit).

1. Create a Python environment with RDkit called 'protein_observed_ms':
	conda create -c rdkit -n protein_observed_ms rdkit
	conda install -n protein_observed_ms biopython scipy matplotlib
	conda activate protein_observed_ms
	pip install cairosvg
