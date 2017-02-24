import math, collections


class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.uniN1=0
    self.train(corpus)
    

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      data = sentence.data
      token1 = data[0].word
      self.unigramCounts[token1] += 1
      for datum in data[1:]:
        token2 = datum.word
        self.unigramCounts[token2] += 1
        self.bigramCounts[(token1,token2)] += 1
        self.total += 1
        token1 = token2
      
    

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    token1 = sentence[0]
    for token2 in sentence[1:]:
      countbi = self.bigramCounts[(token1,token2)]
      count1 = self.unigramCounts[token1]
      count2 = self.unigramCounts[token2]
      
      if countbi > 0:
        score += math.log(countbi) 
        score -= math.log(count1)
      else:
        score += math.log(count2+1) + math.log(.5)
        score -= math.log(self.total+len(self.unigramCounts))
      token1 = token2
    
    return score
    

