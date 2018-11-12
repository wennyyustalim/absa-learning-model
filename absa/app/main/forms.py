# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Required, Length


class ReviewForm(FlaskForm):
    review = StringField(
        'Enter your review', validators=[Required(), Length(1, 128)]
    )
    submit = SubmitField('Submit Review')


# class TodoListForm(FlaskForm):
#     title = StringField(
#         'Enter your review title', validators=[Required(), Length(1, 128)]
#     )
#     submit = SubmitField('Submit Review')
