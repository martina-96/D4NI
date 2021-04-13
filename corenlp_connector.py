import os
os.environ["CORENLP_HOME"] = os.path.abspath(os.getcwd()) + "/stanford-corenlp-4.2.0/"
from stanfordcorenlp import StanfordCoreNLP
import json

class CoreNLPConnector:
    __instance = None

    @staticmethod 
    def getInstance():
      """ Static access method. """
      if CoreNLPConnector.__instance == None:
          CoreNLPConnector()
      return CoreNLPConnector.__instance

    def __init__(self):
      """ Virtually private constructor. """
      if CoreNLPConnector.__instance != None:
          raise Exception("This class is a singleton!")
      else:
          CoreNLPConnector.__instance = self
          self.nlp = StanfordCoreNLP("http://localhost", port=9000, timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
          self.props = {
              'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,coref',
              'pipelineLanguage': 'en'
          }
    
    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        sentence = sentence.lower()
        return json.loads(self.nlp.annotate(sentence, properties= self.props))
    
    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens