install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py
	black Notebooks/*.py

lint:
	pylint --disable=R,C  *.py
	pylint --disable=R,C,W0621,W0104  Notebooks/updateModel.py,updateModelPrediction.py,train.py

updatePrediction:
	python Notebooks/updatePrediction.py

updateModel:
	python Notebooks/updateModel.py
    
updateModel:
	python Notebooks/updateModel.py  

updateModelPrediction:
	python Notebooks/updateModelPrediction.py
  
all: install format lint