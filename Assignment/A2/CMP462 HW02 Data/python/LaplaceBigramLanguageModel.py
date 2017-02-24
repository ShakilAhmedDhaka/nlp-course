import math, collections


class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.N1=0
    self.train(corpus)
    

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      ln = len(sentence)
      for i in range(0,ln-2):
        token1 = sentence.data[i].word
        token2 = sentence.data[i+1].word
        self.bigramCounts[(token1,token2)] = self.bigramCounts[(token1,token2)] + 1
        if self.bigramCounts[(token1,token2)] == 1:
          self.N1+=1
        self.total += 1
    

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    ln = len(sentence)
    for i in range(0,ln-2):
      token1 = sentence[i]
      token2 = sentence[i+1]
      count = self.bigramCounts[(token1,token2)]
      if count > 0:
        score += math.log(count+1)
        score -= math.log(self.total+self.N1)
      else:
        #score = float('-inf') # not smoothed
        score += math.log(1)
        score -= math.log(self.total+self.N1)
    return score
    
