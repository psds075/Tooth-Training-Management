# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, session
import os
import json
import pandas as pd
import requests
import collections
#import cv2
import base64
import numpy as np
import pymongo
#from shapely import geometry
from multiprocessing.pool import ThreadPool
from datetime import datetime
pool = ThreadPool(processes=2)

with open('label_dict.json',encoding = 'utf-8') as json_file:
    data = json.load(json_file)
    LABEL_DICT = data['LABEL_DICT']
    LABEL_DICT_ENG = data['LABEL_DICT_ENG']
    
