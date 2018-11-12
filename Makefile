install:
	pip install -r requirements.txt

run:
	FLASK_APP=./absa/webapp.py flask run

run-prod:
	FLASK_APP=./absa/webapp-production.py flask run --host=0.0.0.0 --port=80
