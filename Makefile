install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py
	black Notebooks/*.py

lint:
	pylint --disable=R,C  *.py
<<<<<<< HEAD
	pylint --disable=R,C,W0621,W0104  Notebooks/updateModel.py,updateModelPrediction.py,train.py
=======
	pylint --disable=R,C,W0621  Notebooks/*.py
>>>>>>> 2fea22f99e9cfdad74e8eb5898e05144036358c7

updatePrediction:
	python Notebooks/updatePrediction.py

updateModel:
	python Notebooks/updateModel.py
    
updateModel:
	python Notebooks/updateModel.py  

updateModelPrediction:
	python Notebooks/updateModelPrediction.py
  
<<<<<<< HEAD
all: install format lint
=======
all: install format lint
>>>>>>> 2fea22f99e9cfdad74e8eb5898e05144036358c7
