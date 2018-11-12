
# -*- coding: utf-8 -*-

import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret key, just for testing'

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    import logging
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )
    logging.getLogger().setLevel(logging.DEBUG)


class ProductionConfig(Config):
    DEBUG = False
    TESTING = True
    WTF_CSRF_ENABLED = True


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
