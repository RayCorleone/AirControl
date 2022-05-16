# -*- coding: utf-8 -*-
"""
    @author:    Ray
    @version:   0.1
	@function:	视图函数
    @problem:   
"""

from air import app
from flask import render_template, jsonify
from air.model import create_bar
import numpy as np

@app.route('/data', methods=['GET'])
def ping_pong():
	bar = create_bar()
	return bar

@app.route('/')
def device():
	context = create_bar()
	return render_template("index.html", context = context)

@app.route('/capture', methods=['GET'])
def capture():
	context = create_bar()
	return render_template("index.html", context = context)


@app.route('/update', methods=['POST'])
def update():
    context = create_bar()
    print("JSONIFY")
    return jsonify('', render_template("myDiv.html", context = context))

