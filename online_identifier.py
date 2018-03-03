#######################################################
# Online Object / Face identifier
# @author: Christian Reichel
# Version: 0.2
# -----------------------------------------------------
# Sends the image as numpy array to pretrained network
# to detect objects (especially faces)
#######################################################

# IMPORTS
import requests		# sending requests
import json 		# json data actions
import numpy as np  # checking data type

# Parameters
server_url = 'http://letsfaceit.quving.com/api/predict'

class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def identify(image_np):
    
    data = {}
    data["image"]= json.dumps(image_np.tolist())
    data["mode"]= "predict"
    response = requests.post(server_url, json = data)
    return response.json()["label"], response.json()["probability"]
