# D4NI - Data structured for neural network instrument
### D4NI è un servizio web che sfrutta un algoritmo intelligente che ha lo scopo di creare dati strutturati per corpus.Permette di mostrare diversi tipi di ooutput che un'analisi testuale può fornire ed effettuare il download dei dati nella forma più comoda per essere usati con Spektral.


## Getting started

Before usage: 
- Download Stanford library from: http://nlp.stanford.edu/software/stanford-corenlp-4.2.0.zip
- Download gloVe models from: http://nlp.stanford.edu/data/glove.6B.zip
- Create 4 files named respectively: - PartOfSpeech50.txt, PartOfSpeech100.txt, PartOfSpeech200.txt, PartOfSpeech300.txt
    These files must containt a list of all Part of speech tag known and every one of them must have a unique vector respectively of lenght: 50, 100, 200, 300.
    The content of the vectors elements is up to the user as far as thet are unique.
    Unfortunately due to gituhub restriction on file size they will be not uploaded in this repository.

Stanford library must be under the directory of the project.
gloVe files must be under the directory ./static/glove_files
Part of speech files must be under the directory ./static/part_of_speech

Use this command to install all the dependecies
`pip install -e .`

Run the nlp server of standford!!
`java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000`

Run the server!!
`flask run`

You can use this command for starting the server in development mode
`export FLASK_ENV=development`



