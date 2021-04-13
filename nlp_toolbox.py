import numpy as np
from corenlp_connector import CoreNLPConnector
from nltk.tree import Tree

class NlpToolBox:
    text = ''
    glove_model = {}
    lenght = 0,
    one_hot_model = {}
    raw_annotated = {}
    nltk_trees = {}
    str_trees = {}
    adjacency_matrixes = {}
    feature_matrixes = {}

    def __init__(self, corpus, glove_filename):
        self.text = corpus
        self.glove_model, self.lenght = NlpToolBox.load_glove_model(glove_filename)
        self.one_hot_model = NlpToolBox.get_one_hot_model(self.lenght)
        self.raw_annotated = NlpToolBox.nlp_annotate(corpus) #raw data
        self.nltk_trees, self.str_trees, self.label_trees = NlpToolBox.get_trees(self.raw_annotated) #trees
        self.n_nodes = NlpToolBox.get_numerosity(self.nltk_trees) #array of numerosities
        self.adjacency_matrixes = NlpToolBox.adjacency_matrixes(self.n_nodes, self.nltk_trees) #array of matrixes
        self.feature_matrixes = NlpToolBox.feature_matrixes(self.n_nodes, self.nltk_trees, self.glove_model, self.one_hot_model) #array of matrixes

    #Setup models
    @staticmethod
    def load_glove_model(glove_filename):
        with open(glove_filename, encoding="utf8") as f: content = f.readlines()
        model = {}
        num = 0
        for line in content:
            splitLine = line.split() #word
            word = splitLine[0]  #every single word in the document
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        num = model['the'].size #i use a model of the word 'the' becouse anything in the file glove have the same size
        return model, num

    @staticmethod
    def get_one_hot_model(lenght):
        if lenght == 50 :
            one_hot_model = NlpToolBox.one_hot_encoding("./static/part_of_speech/PartOfSpeech50.txt")
        elif lenght == 100 :
            one_hot_model = NlpToolBox.one_hot_encoding("./static/part_of_speech/PartOfSpeech100.txt")
        elif lenght == 200 :
            one_hot_model = NlpToolBox.one_hot_encoding("./static/part_of_speech/PartOfSpeech200.txt")
        elif lenght == 300 :
            one_hot_model = NlpToolBox.one_hot_encoding("./static/part_of_speech/PartOfSpeech300.txt")
        return one_hot_model

    @staticmethod
    def one_hot_encoding(text):
        with open(text, encoding="utf8" ) as f: content = f.readlines()
        model = {}
        num = 0
        for line in content:
            splitLine = line.split() #word
            word = splitLine[0]
            embedding = np.array([float(val)for val in splitLine[1:]])
            model[word] = embedding
        return model
    
    #Main StandfordNlp method
    def nlp_annotate(text):
        raw_annotated = CoreNLPConnector.getInstance().annotate(text)
        return raw_annotated
    
    #Parsing StandfordNlp result
    @staticmethod
    def get_trees(raw_annotated):
        nltk_tree = []
        str_tree = []
        label_trees = []
        for i in raw_annotated['sentences']:
            tmp_str = i['parse']
            tmp_str = tmp_str.replace("(. .)", "")
            tmp_nltk = Tree.fromstring(tmp_str)
            tmp_labels = np.array([], dtype=object)
            label_trees.append(NlpToolBox.tree_labels(tmp_nltk, tmp_labels))
            str_tree.append(tmp_str)
            nltk_tree.append(tmp_nltk)
        return nltk_tree, str_tree, label_trees
    
    #For single tre mehods
    @staticmethod
    def get_numerosity_single_tree(tree,temp = 0): #of a single tree
        sum = 1
        h = 0
        count_intermediate = temp
        for node in tree:
            if type(node) is not Tree:
                return sum + 1
            else:
                count_intermediate += 1
                h = NlpToolBox.get_numerosity_single_tree(node,count_intermediate)
                sum += h
        return sum

    @staticmethod
    def single_adjacency_matrix(mat,tree,temp=0):
        selector = -1
        count = temp
        t = temp
        for node in tree:
            selector += 1
            count += 1
            if type(node) is not Tree:
                mat[count-1][count] = 1
                mat[count][count-1] = 1
                return mat,count
            else:
                if selector == 0 : #control the node in the for each
                    mat[count][count-1] = 1
                    mat[count-1][count] = 1 
                else:
                    mat[t][count] = 1
                    mat[count][t] = 1
                mat,count = NlpToolBox.single_adjacency_matrix(mat,node,count)
        return mat,count

    @staticmethod
    def single_feature_matrix(mat,tree,model,Hot):
        for node in tree:
            if type(node) is not Tree: # if it is a leaf
                mat = np.append([mat], [model[node]], axis=0) if mat.shape == model[node].shape else np.append(mat, [model[node]], axis=0)
                return mat
            else: 
                mat = np.append([mat], [Hot[node.label()]], axis=0) if mat.shape == Hot[node.label()].shape else np.append(mat, [Hot[node.label()]], axis=0)
                mat = NlpToolBox.single_feature_matrix(mat,node,model,Hot)
        return mat

    #For all trees methods
    @staticmethod
    def get_numerosity(trees):
        numerosity = []
        for tree in trees:
            n = NlpToolBox.get_numerosity_single_tree(tree)
            numerosity.append(n)
        return numerosity

    @staticmethod
    def adjacency_matrixes(n_nodes, trees):
        adjacency_matrixes = []
        i = 0
        for tree in trees:
            start_matrix = np.zeros([n_nodes[i],n_nodes[i]])
            matrix, trash = NlpToolBox.single_adjacency_matrix(start_matrix, tree)
            adjacency_matrixes.append(matrix)
            i += 1
        return adjacency_matrixes

    @staticmethod
    def feature_matrixes(n_nodes, trees, glove_model, one_hot_model):
        feature_matrixes = []
        i = 0
        for tree in trees:
            start_matrix = np.array(one_hot_model['ROOT'])
            matrix = NlpToolBox.single_feature_matrix(start_matrix, tree, glove_model, one_hot_model)
            i += 1
            feature_matrixes.append(matrix)
        return feature_matrixes

    @staticmethod
    def tree_labels(nltk_tree, array_labels):
        for node in nltk_tree:
            if type(node) is not Tree:
                array_labels = np.append(array_labels, node)
                return array_labels
            else: 
                array_labels = np.append(array_labels, node.label())
                array_labels = NlpToolBox.tree_labels(node, array_labels)
        return array_labels