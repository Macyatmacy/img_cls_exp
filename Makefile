install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py
	black Notebooks/*.py

lint:
	pylint --disable=R,C  *.py
	pylint --disable=R,C,W0621,W0104,W0102  Notebooks/updateModel.py
	pylint --disable=R,C,W0621,W0104,W0102  Notebooks/updateModelPrediction.py
	pylint --disable=R,C,W0621,W0104,W0102  Notebooks/train.py


updatePrediction:
	python Notebooks/updatePrediction.py

updateModel:
	python Notebooks/updateModel.py 

updateModelPrediction:
	python Notebooks/updateModelPrediction.py
  

all: install format lint
