import math, collections


class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.tetragramCounts = collections.defaultdict(lambda: 0)
    self.trigramCounts = collections.defaultdict(lambda: 0)
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
      token2 = data[1].word
      token3 = data[2].word
     
      self.unigramCounts[token1] += 1
      self.unigramCounts[token2] += 1
      self.unigramCounts[token3] += 1
      
      self.bigramCounts[(token1,token2)] += 1
      self.bigramCounts[(token2,token3)] += 1

      self.trigramCounts[(token1,token2,token3)] += 1
      
      for datum in data[3:]:
        token4 = datum.word
        self.unigramCounts[token4] += 1
        self.bigramCounts[(token3,token4)] += 1
        self.trigramCounts[(token2,token3,token4)] += 1
        self.tetragramCounts[(token1,token2,token3,token4)] += 1
        self.total += 1
        token1 = token2
        token2 = token3
        token3 = token4
      self.total += 3

      
      
    

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    token1 = sentence[0]
    token2 = sentence[1]
    
    #score +=  math.log(self.unigramCounts[token2])
    #score -= math.log(self.total+len(self.unigramCounts))
    for token3 in sentence[2:]:
      countri = self.trigramCounts[(token1,token2,token3)]
      countbi1 = self.bigramCounts[(token1,token2)]
      countbi2 = self.bigramCounts[(token2,token3)]
      count1 = self.unigramCounts[token1]
      count2 = self.unigramCounts[token2]
      count3 = self.unigramCounts[token3]
      
      if countri > 0:
        score += math.log(countri)
        score -= math.log(self.total-3)
      elif countbi2 > 0:
        score += math.log(countbi2) + math.log(.4)
        score -= math.log(self.total-2)
      else:
        score += math.log(count3+1) + math.log(.4)
        score -= math.log(self.total) + math.log(len(self.unigramCounts))

      token1 = token2
      token2 = token3
    
    return score
    

