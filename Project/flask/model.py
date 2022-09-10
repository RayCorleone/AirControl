# -*- coding: utf-8 -*-
"""
    @author:    Ray
    @version:   0.1
	@function:	数据模型
    @problem:   
"""

import plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json

def create_bar():
	pyplt = py.offline.plot
	#---以下因为图形和数据不同而不同，以下需要修改---
	N = 40
	x = np.linspace(0, 1, N)
	y = np.random.randn(N)
	df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
	trace0 = go.Bar(
		x=df['x'], # assign x as the dataframe column 'x'
		y=df['y']
	)
	data = [trace0]
	layout = go.Layout(
		title='xxx可视化作图',
	)
	fig = go.Figure(data=data, layout=layout)
	jsfig = fig.to_json()
	return jsfig

