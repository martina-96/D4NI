from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from nlp_toolbox import NlpToolBox
from spacy import displacy
from helpers import *
import zipfile

app = Flask(__name__)

app.secret_key = b'chiavesegreta1234'
app.config['dataset_path'] = 'dataset/'

# app.logger.info('%s logged in successfully', user.username)

@app.route('/', methods = ["GET"])
def index():
    return render_template("form.html")

@app.route('/', methods = ["POST"])
def init_transaction():
    #delete existing files 
    remove_files(app.config['dataset_path'])
    #request form data
    textarea = format(request.form['testo'])
    uploaded_file = request.files["file"]
    glove_option = request.form.get('Glove')
    output_option = request.form.get('Output')

    #defining variables
    uploaded_file_in_str = uploaded_file.read().decode("utf-8")
    glove_file_path = choose_glove_file(glove_option)

    if uploaded_file_in_str and allowed_file_type(uploaded_file.filename):
        corpus = uploaded_file_in_str
    elif textarea:
        corpus = textarea
    else:
        return render_template("form.html", error="Both file and text can't be blank")

    toolbox_data = NlpToolBox(corpus, glove_file_path)
    adjacency_matrixes = toolbox_data.adjacency_matrixes
    feature_matrixes = toolbox_data.feature_matrixes
    spacy_format = to_spacy_format(toolbox_data.raw_annotated)
    syntree_format = syntree(toolbox_data.str_trees)
    session['SpacyFormat'] = spacy_format
    session['SyntreeFormat'] = syntree_format

    if output_option == 'DS':

        #DELETE ALL FILES IN STATIC/DATASET BEFORE CONTINUE
        create_dataset(feature_matrixes, adjacency_matrixes, toolbox_data.label_trees, app.config['dataset_path'])
        return send_from_directory(
            app.config['dataset_path'],
            filename='D4NI_dataset.zip',
            as_attachment=True
        )
        #return Response(create_dataset(feature_matrixes, adjacency_matrixes, toolbox_data.label_trees, 'dataset'), mimetype='multipart/x-zip')
    else:
        return redirect(url_for(render_option(output_option)))

    return render_template("form.html")

@app.route("/graph-result")
def display_graph():
    spacy_format = session['SpacyFormat']
    svgs = spacyRender(spacy_format)
    return render_template("display_graph.html", graph_data=svgs)

@app.route("/tree-result")
def display_tree():
    syntree_format = session['SyntreeFormat']
    return render_template("display_tree.html", tree_data=syntree_format)