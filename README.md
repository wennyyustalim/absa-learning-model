# How Good is My Hotel?

## Set up
### Docker

    docker-compose build
    docker-compose up

Runs on http://localhost:8000/

### Manual
Run:
    pip install -r requirements.txt
    FLASK_APP=webapp.py flask run

Add some play data.
    pip install -r test-requirements.txt
    flask fill_db

Runs on http://localhost:5000/