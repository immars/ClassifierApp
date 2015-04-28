# -*- coding: utf-8 -*-
# script visualizing weights of a layer using PCA

from caffe import *
from parser import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

model=['model/maxmap_deploy.prototxt','model/bvlc_maxmap_iter_250000.caffemodel','loss1/classifier']
# model=['model/googlenet_deploy.prototxt','model/bvlc_googlenet_iter_1000000.caffemodel','loss3/classifier']

net=Net(model[0],model[1],TEST)
vec=net.params[model[2]][0].data

vec2=np.vstack((vec, vec, vec, vec))
pca=mlab.PCA(vec2)
lm=LabelMap()
lm.load('model/term.txt')
vec2=[pca.project(v)[:2] for v in vec]

matplotlib.rc('font', family='NanumGothic')

for i in range(len(vec2)):
    v=vec2[i]
    plt.text(v[0],v[1], lm.id2labels[i].names[0])
    plt.plot(v[0],v[1])
plt.show()

