def split_to_sentences(dts):
  """
  Split each line of dataset to sentences if possible.
  Return: dataset[]
  """
  new_dts = []
  if type(dts) is str:
    new_dts = dts.replace('\n')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    new_dts = tokenizer.tokenize(new_dts)
    return new_dts
  for i in range(len(dts)):
    tmp = re.split(r'[.?!]\s*', dts[i])
    for sent in tmp:
      if sent:
        new_dts.append(sent)
  return new_dts

def remove_html(dts):
  """
  Remove HTML tags from dataset.
  Eg: <br>, <h1>,...
  Return: dataset[]
  """
  cleanr = re.compile('<.*?>')
  for i in range(len(dts)):
    dts[i] = re.sub(cleanr, '', dts[i])
  return dts

def lowercase(dts):
  """
  Convert each sentence in dataset into lowercase form.
  Return: dataset[]
  """
  for i in range(len(dts)):
    dts[i] = dts[i].lower()
  return dts

def remove_punctuations(dts):
  """
  Remove punctuations from dataset.
  Return: dataset[]
  """
  for i in range(len(dts)):
    for punc in string.punctuation:
      dts[i] = dts[i].replace(punc, '')
  return dts

def remove_special_characters(dts):
  """
  Remove special characters from dataset.
  Eg: @ # $ % ^ blank...
  Return: dataset[]
  """
  for i in range(len(dts)):
   dts[i] = re.sub('[^A-Za-z0-9 ]+', '', dts[i])
   dts[i] = ' '.join(dts[i].split())
  return dts

def drop_short_sentences(dts):
  """
  Remove sentences that only have 1 or 2 word(s),
  Return: dataset[]
  """
  new_dts = []
  for cmt in dts:
    if len(cmt.split(' ')) > 1:
      new_dts.append(cmt)
  return new_dts

def padding(dts, ngrams=2):
  """
  Add <s> and </s> to each sentence according to n.
  Return: dataset[]
  """
  for i in range(len(dts)):
    dts[i] = '<s> '*ngrams  + dts[i] + ' </s>'*ngrams
  return dts

def preprocess(dts):
  """
  Preprocessing input dataset:
    - Remove HTML tags
    - Lowercase
    - Remove punctuations
    - Remove special characters
    - Padding
  Return: dataset[]
  """
  dts = remove_html(dts)
  dts = lowercase(dts)
  dts = remove_punctuations(dts)
  dts = remove_special_characters(dts)
  dts = padding(dts)
  return dts

def build_vocabulary(dts):
  """
  Build vocabulary from input dataset.
  Return: vocabulary[]
  """
  vocab = set()
  for sent in dts:
    for word in sent.split(' '):
      if word:
        vocab.add(word)
  return list(vocab)


def count_sequence(seq, dts):
  """
  Count occurrences of input sequence in entire dataset.
  Return: int
  """
  n = 0
  for sent in dts:
    n += sent.count(seq)
  return n

def build_trigram_model(dts):
  """
  Build trigram model from dataset.
  The model is a dictionary with each item is (trigram sequence): probability.
  Return: model{}
  """
  model = {} # each item is (wn-2, wn-1, wn): probability]
  for sent in dts:
    tokens = sent.split(' ')
    for i in range(2, len(tokens)):
      wn = tokens[i]
      wn_1 = tokens[i-1]
      wn_2 = tokens[i-2]
      
      # calculate probability with Laplace smoothing
      C_tri = count_sequence(wn_2 + ' ' + wn_1 + ' ' + wn, dts)
      C_bi = count_sequence(wn_2 + ' ' + wn_1, dts)
      prob = (C_tri + 1) / (C_bi + V)

      # save result to model
      model[(wn_2, wn_1, wn)] = prob
  
  return model

def generate1(seed_text, model):
  """
  Generate 1 next word that has highest probability on seed_text with probability value.
  Return: (string, float)
  """
  text = preprocess([seed_text])[0]
  text = text.split(' ')
  wn_1 = text[-3]
  wn_2 = text[-4]
  
  predicted_word = 'UNK'
  predicted_prob = 0

  for key, value in model.items():
    if wn_2 == key[0] and wn_1 == key[1]:
      if predicted_prob < value:
        predicted_word = key[2]
        predicted_prob = value
  return (predicted_word, predicted_prob)

def generate(seed_text, model, n=1):
  """
  Generate n next words and probability of whole generated sequence
  until meet <UNK> or </s> symbol.
  Return: (string, float)
  """
  tmp = ''
  tmp2 = seed_text
  sent_prob = 1
  for i in range(n):
    text, prob = generate1(tmp2 + tmp, model)
    if text == 'UNK' or text == '</s>':
      break 
    tmp = tmp + ' ' + text
    sent_prob *= prob 
  return seed_text + tmp, sent_prob

def evaluate(test_set, model, vocabulary):
  """
  Calculate the Perplexity of N-gram language model.
  Return: float
  """
  V = len(vocabulary)
  PP = 1
  for sent in test_set:
    tokens = sent.split(' ')
    for i in range(2, len(tokens)):
      wn = tokens[i]
      wn_1 = tokens[i-1]
      wn_2 = tokens[i-2]

      # Find in model
      found = False
      for key, value in model.items():
        if (wn_2, wn_1, wn) == key:
          PP *= value
          found = True
          break
      
      # If not in model:
      if found == False:
        PP *= 1 / V
      
      PP = PP**(-1/N)

  return PP