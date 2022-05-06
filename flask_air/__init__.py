# -*- coding: utf-8 -*-
"""
    @author:    Ray
    @version:   1.0
	@function:	实例化程序
    @problem:   无
"""

from flask import Flask
from flask_bootstrap import Bootstrap4

app = Flask('air')
bootstrap = Bootstrap4(app)

from air import view