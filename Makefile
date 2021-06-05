install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py
	black Notebooks/*.py

lint:
	pylint --disable=R,C  *.py
	pylint --disable=R,C,W0621  Notebooks/*.py

all: install format lint