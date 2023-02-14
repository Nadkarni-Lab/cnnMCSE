sync-results:
	scp -r gulamf01@minerva.hpc.mssm.edu:/sc/arion/projects/EHR_ML/gulamf01/cnnMCSE/output .


.PHONY: clean install data requirements

init:
	pip install -U pip
	pip install -U setuptools
	pip install -r requirements.txt
	python setup.py install

update:
	python setup.py install


clean:
	python setup.py clean --all