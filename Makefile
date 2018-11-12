install:
	pip install -r requirements.txt

run:
	FLASK_APP=./src/absa-webapp/webapp.py flask run

