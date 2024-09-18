import numpy as np
def find_entropy(df):
  Class = df.keys()[-1]
  values = df[Class].unique()
  entropy = 0
  for value in values:
    prob = df[Class].value_counts()[value]/len(df[Class])
    entropy += -prob * np.log2(prob)
  return np.float(entropy)
# Find entropy attribute
def find_entropy_attribute(df, attribute):
  Class = df.keys()[-1]
  target_values = df[Class].unique()
  attribute_values = df[attribute].unique()
  avg_entropy = 0
  for value in attribute_values:
    entropy = 0
    for value1 in target_values:
      num = len(df[attribute][df[attribute] == value][df[Class] == value1])
      den = len(df[attribute][df[attribute] == value])
      prob = num/den
      entropy += -prob * np.log2(prob + 0.000001)
    avg_entropy += (den/len(df))*entropy
  return np.float(avg_entropy)
     

# Find Winner
def find_winner(df):
  IG = []
  for key in df.keys()[:-1]:
    IG.append(find_entropy(df) - find_entropy_attribute(df, key))
  return df.keys()[:-1][np.argmax(IG)]

def get_subtable(df, attribute, value):
  return df[df[attribute] == value].reset_index(drop = True)
     

def buildtree(df, tree = None):
  node = find_winner(df)
  attvalue = np.unique(df[node])
  Class = df.keys()[-1]
  if tree is None:
    tree = {}
    tree[node] = {}
  for value in attvalue:
    subtable = get_subtable(df,node,value)
    Clvalue, counts = np.unique(subtable[Class], return_counts = True)
    if len(counts) == 1:
      tree[node][value] = Clvalue[0]
    else:
      tree[node][value] = buildtree(subtable)
  return tree
     

import pandas as pd
df = pd.read_csv('weather.csv')
     
tree = buildtree(df)
import pprint
pprint.pprint(tree)
     