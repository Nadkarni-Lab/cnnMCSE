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

interactive:
	bsub -P acc_EHR_ML -q interactive -n 8 -W 60 -R span[hosts=1] -Is /bin/bash


tensorboard:
	ml purge
	ml python/3.8.2
	unset PYTHONPATH
	source /sc/arion/projects/EHR_ML/gulamf01/archive/aICP/src/venv/bin/activate
	tensorboard --logdir /sc/arion/projects/EHR_ML/gulamf01/aICP/src/2_optimize_model/lightning_logs