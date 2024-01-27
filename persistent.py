"""
This module gathers persistent data from the horde when Invoke is first run.
"""
from invokeai.backend.util.logging import info, debug, warning, error
# Load the local userconfig.cfg file
import configparser
import os
import shutil

config = configparser.ConfigParser()
#get path to the local folder
path = os.path.dirname(os.path.realpath(__file__))
#join the path to the config file

# Check if userconfig.cfg exists, if not, create it as a copy of userconfig.TEMPLATE.cfg
if not os.path.exists(os.path.join(path, 'userconfig.cfg')):
    info('Creating userconfig.cfg from template.')
    shutil.copy(os.path.join(path, 'userconfig.TEMPLATE.cfg'), os.path.join(path, 'userconfig.cfg'))

config.read(os.path.join(path, 'userconfig.cfg'))
APIKEY = config.get('settings', 'apikey')
MINIMUM_WORKERS = config.getint('settings', 'minimum_workers')


import requests
import json

BASEURL = 'https://stablehorde.net/api/v2/'
CLIENT_AGENT = 'InvokeAI:0.1:@dunkeroni'



#get list of models from /status/models
MODEL_LIST = []

""" Model Format:
[
  {
    "name": "string",
    "count": 0,
    "performance": 0,
    "queued": 0,
    "jobs": 0,
    "eta": 0,
    "type": "image"
  }
]
"""
try:
    url = BASEURL + 'status/models'
    headers = {'Client-Agent': CLIENT_AGENT}
    params = {'type': 'image', 'min_count': MINIMUM_WORKERS}
    r = requests.get(url, headers=headers, params=params)
    MODEL_LIST = r.json()
    MODEL_LIST.sort(key=lambda x: x['count'], reverse=True)
    info(f"Fetched {len(MODEL_LIST)} models from the horde.")
except:
    error('Could not get model list from the horde.')
