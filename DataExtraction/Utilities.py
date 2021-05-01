
import json
from splinter import Browser
from os.path import dirname, abspath
from itertools import chain
import requests
import shutil
import pdftotext
from six.moves.urllib.request import urlopen
import io


def save_json(data, file = 'data.json', sort_keys = True):
    with open('Data/' + file, 'w') as file:
        json.dump(data, file, indent = 4, sort_keys = sort_keys)

def load_json(file = 'data.json'):
    with open('Data/' + file) as file:
        return json.load(file)

def chromeBrowser(headless = True):
    driver_path = dirname(abspath(__file__)) + '/chromedriver'
    return Browser('chrome', executable_path = driver_path, headless = headless)

def flat_map(f, items):
    return chain.from_iterable(map(f, items))

def download_pdf(from_url, to_file_path):
    r = requests.get(from_url, stream = True)
    r.raw.decode_content = True
    with open(to_file_path, 'wb') as f:
        shutil.copyfileobj(r.raw, f) 

def read_pdf_from_url(url):
    remote_file = urlopen(url).read()
    memory_file = io.BytesIO(remote_file)
    pdf = pdftotext.PDF(memory_file)
    
    return '\n\n'.join(pdf)

def read_pdf_from_file(file_path):
    with open(file_path, 'rb') as f:
        try:
            pdf = pdftotext.PDF(f)
        except:
            return None
    
    return '\n\n'.join(pdf)
