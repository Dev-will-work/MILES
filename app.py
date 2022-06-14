from flask import redirect, url_for, render_template, abort, send_from_directory, send_file, jsonify, request
from flask import Flask

from gevent.pywsgi import WSGIServer
import threading
import random
import json
import time
import os
import io
import sys
import getopt

from simplifier import simplifier
from simplifier import models
from simplifier.config import *

#Create app
app = Flask(__name__,static_url_path='')
app.debug = False

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/ai_request/', methods=['GET', 'POST'])
def ai_request():
    if request.method == 'GET':
        text = request.args.get('q')
        id = request.args.get('id')
        
        if id not in data_map:
            data_map[id] = UserData("en", False)
            data_map[id].set_embeddings(models.load_id_embeddings("en", id))
        
        if text.startswith("language:") and not data_map[id].disable_embeddings:
            data_map[id].set_lang(text[9:])
            #config.lang = text[9:]
            if data_map[id].lang in supported_langs:
                data_map[id].embeddings = models.load_id_embeddings(data_map[id].lang, id)
                
        if text == "_without_embeddings":
            data_map[id].set_disable(True)
        if text == "_with_embeddings":
            data_map[id].set_disable(False)
            
        print(data_map[id])
        
        output = simplifier.simplify_id_text(text, id, bold_highlight=True)
        return jsonify({"output": output})
    
    if request.method == 'POST':
        text = request.form.get('input')
        
        if text.startswith("language:") and not disable_embeddings:
            lang = text[9:]
            if lang in supported_langs:
                models.embeddings = models.load_embeddings(lang)
                
        if text == "_without_embeddings":
            disable_embeddings = True
        if text == "_with_embeddings":
            disable_embeddings = False
            
        output = simplifier.simplify_text(text, bold_highlight=True)
        return jsonify({"output": output})


if __name__ == "__main__":

    if len(sys.argv) > 1:
        
        argv = sys.argv[1:]

        try:
            opts, args = getopt.getopt(argv, "l:")
        except:
            print("Error!")

        # Get language if passed.
        for opt, arg in opts:
            if opt in ['-l', '--language']:
                lang = arg

    # Check if language is supported and attempt to load embeddings.
    if lang in supported_langs:
        models.embeddings = models.load_embeddings(lang)

    print("\nStarting MILES Flask server...")
    http_server = WSGIServer(('0.0.0.0', 80), app)    
    print("\nLoaded as HTTP Server on port 80, running forever:")
    http_server.serve_forever()

