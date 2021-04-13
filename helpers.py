#to render spacy
from spacy import displacy
import urllib.parse
#to create dataset
from tempfile import TemporaryFile, TemporaryDirectory
import numpy as np
import zipfile
import os
#import requests
#from bs4 import BeautifulSoup as bs #usefull to generate tree images (not links to websites as it does now)

ALLOWED_EXTENSIONS = {'txt'}

def choose_glove_file(option):
    switcher = {
                "G1": "./static/glove_files/glove.6B.50d.txt",
                "G2": "./static/glove_files/glove.6B.100dTODO.txt",
                "G3": "./static/glove_files/glove.6B.200dTODO.txt",
                "G4": "./static/glove_files/glove.6B.300dTODO.txt"
                }

    return switcher.get(option, lambda :'Invalid option')

def allowed_file_type(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def to_spacy_format(rawAnnotated):
  spacyFormat = []
  for sentence in rawAnnotated['sentences']:
    words_list = []
    for token in sentence['tokens']:
      if not token['word'] == '.':
        tag = token['pos']
        word_dict = {
          "text": token['word'],
          "tag": tag
        }
        words_list.append(word_dict)
    arcs_list = []
    for edge in sentence['enhancedDependencies']:
      if (edge['dep'] != 'ROOT') and not ((edge['governorGloss'] == '.') or (edge['dependentGloss'] == '.')):
        if edge['governor']<edge['dependent']:
          start = edge['governor']-1
          end = edge['dependent']-1
          direction = "right"
        else:
          start = edge['dependent']-1
          end = edge['governor']-1
          direction = "left"
        label = edge['dep']
        arc_dict = {
          "start": start,
          "end": end,
          "label": label,
          "dir": direction
        }
        arcs_list.append(arc_dict)
    sentence_dict = {
      "words": words_list,
      "arcs": arcs_list
    }
    spacyFormat.append(sentence_dict)
  return spacyFormat

def spacyRender(jsonSpacy):
  if jsonSpacy:
    svgs_encoded = []
    options = {"compact": False, "manual": True, "distance": 120, "word_spacing": 20}
    for sentence in jsonSpacy:
      svg = displacy.render(sentence, style="dep", manual=True, options=options)
      svg_encoded = urllib.parse.quote(svg)
      svgs_encoded.append(svg_encoded)
    return svgs_encoded
  else: 
    return False

def syntree(trees):
  link_str = 'http://mshang.ca/syntree/?i='
  for tree in trees:
    encoded_tree = tree.replace("\n", "")
    encoded_tree = encoded_tree.replace("(", "[")
    encoded_tree = encoded_tree.replace(")", "]")
    encoded_tree = encoded_tree.replace(" ", "%20")
    link_str = link_str + encoded_tree
    #if you find a method to rendere the js on the website
    """r = requests.get(link_str)
    content = bs(r.text)
    img_content = content.find(id="image-goes-here")"""
  link_str = link_str.replace("][ROOT", "")
  return link_str

def render_option(option):
  switcher = {
                "GR": "display_graph",
                "TR": "display_tree"
              }
  return switcher.get(option, lambda :'Invalid option')

def create_dataset(feature_matrixes, adjacency_matrixes, label_trees, path):
  for i in range(len(feature_matrixes)):
      x = feature_matrixes[i]
      a = adjacency_matrixes[i]
      e = None
      y = label_trees[i]
      np.savez(f'{path}graphs/graph_{i}.npz', x=x, a=a, e=e, y=y)

  compression = zipfile.ZIP_DEFLATED
  zf = zipfile.ZipFile(f"{path}D4NI_dataset.zip", mode="w")
  for files in os.walk(f'{path}graphs/'):
    for file_name in files[2]:
      zf.write(f'{path}graphs/' + file_name, file_name, compress_type=compression)
  zf.close()
  return True

def remove_files(path):
  for files in os.walk(f'{path}'):
    for file_name in files[2]:
      os.remove(f'{files[0]}/{file_name}')
  return True
