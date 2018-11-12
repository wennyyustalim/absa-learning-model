# -*- coding: utf-8 -*-

from flask import render_template, redirect, request, url_for

from app.main import main, aspect_extractor, sentiment_extractor
from app.main.forms import ReviewForm


@main.route('/',  methods=['GET', 'POST'])
def index():
    sentence_aspects = None
    if request.method == 'GET':
        form = ReviewForm()
        if form.validate_on_submit():
            return redirect(url_for('main.index'))
        return render_template('index.html', form=form, sentence_aspects=sentence_aspects)
    else:
        form = ReviewForm(review=request.form.get('review'))

        if form.validate():
            aspects = aspect_extractor.predict(
                form.review.data.split("."))
            sentence_aspects = sentiment_extractor.predict(aspects)
            return render_template('index.html', form=form, sentence_aspects=sentence_aspects)
        return redirect(url_for('main.index'))
