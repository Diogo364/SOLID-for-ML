.ONESHELL:
SHELL = /bin/zsh

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate solid-ml; conda activate solid-ml

create-env:
	conda env create -f environment.yml;

get-data:
	@echo "Downloading Data"
	$(CONDA_ACTIVATE)
	python -m setup.get_wine_data;
	@echo "Done!"

setup: create-env get-data

delete-env:
	$(CONDA_ACTIVATE)
	conda deactivate
	conda env remove --name solid-ml;