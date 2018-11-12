# -*- coding: utf-8 -*-

from flask import Blueprint

from app.aspect_extractor.aspect_extractor import AspectExtractor
from app.sentiment_extractor.sentiment_extractor import SentimentExtractor

main = Blueprint('main', __name__)

aspect_extractor = AspectExtractor()
aspect_extractor.load_model("./model/aspect_extractor.mdl",
                            "./model/aspect_extractor_mlb.mdl")

sentiment_extractor = SentimentExtractor()
sentiment_extractor.load_model("./model/sentiment_extractor.mdl")

from . import views