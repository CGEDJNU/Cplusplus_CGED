#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

try: 
  import xml.etree.cElementTree as ET 
except ImportError: 
  import xml.etree.ElementTree as ET 

from pyltp import Segmentor
from pyltp import Postagger

import random

def getMaximumInterval(l, r, t):
  n = len(l)
  mx = 0
  for i in range(1 << n):
    cnt = 0
    flag = 0
    for j in range(n):
      cnt += i >> j & 1
    for j in range(n):
      for k in range(j + 1, n):
        if (i >> j & 1) and (i >> k & 1) and (max(r[j], r[k]) - min(l[j], l[k]) + 1 < r[j] - l[j] + r[k] - l[k] + 2):
          flag = 1
    if flag == 0 and cnt > mx:
      mx = cnt
      mask = i
  resl = []
  resr = []
  rest = []
  #print mask, '~'
  for i in range(n):
    if (mask >> i & 1):
      resl.append(l[i])
      resr.append(r[i])
      rest.append(t[i])
  return resl, resr, rest, n - len(resl)

def wordToChar(word, pos):
  resw = []
  resp = []
  for i in range(len(word)):
    for j in range(len(word[i])):
      resw.append(word[i][j])
      resp.append(('B-' if j == 0 else 'I-') + pos[i])
  return resw, resp

  
#Divide the data set into training set and development set.

def main():
  #parser = ET.XMLParser(encoding="utf-8")
  #tree = ET.fromstring('train.xml')
  input_file = sys.argv[1]
  train_file = sys.argv[2]
  test_file = sys.argv[3]
  test_size = int(sys.argv[4])
  segmentor = Segmentor()  # 初始化实例
  segmentor.load('./cws.model')  # 加载模型
  postagger = Postagger() # 初始化实例
  postagger.load('./pos.model')  # 加载模型

  output_list = []
  tree = ET.parse(input_file)
  root = tree.getroot()
  cnt = 0
  tt = 0
  tot = 0
  for doc in root.findall('DOC'):
    output = []
    #if cnt > 1: break
    sentence = doc.find('TEXT').text[1:-1]
    if (len(sentence.replace(' ', '')) != len(sentence)):
      tot += 1
      print (sentence)
    sentence = sentence.replace(' ', '')
    word = segmentor.segment(sentence.encode('utf-8'))  # 分词
    #word = []
    #for i in range(len(sentence)):
    #  word.append(sentence[i].encode('utf-8'))
    pos = postagger.postag(word)  # 词性标注
    tag = []
    
    word = list(word)
    pos = list(pos)
    for i in range(len(word)):
      word[i] = word[i].decode('utf-8')
      pos[i] = pos[i].decode('utf-8')

    word, pos = wordToChar(word, pos)
    for i in range(len(word)):
      tag.append('O')
    l = []
    r = []
    t = []
    for error in doc.findall('ERROR'):
      st = int(error.get('start_off'))
      ed = int(error.get('end_off'))
      tp = error.get('type')
      cur = 1
      flag = 0
      for i in range(len(word)):
        if (tp == 'M' and st != ed and st <= cur + len(word[i]) and st > cur and st != cur + len(word[i])) or \
          (tp != 'M' and ((st > cur and st <= cur + len(word[i]) - 1) or (ed >= cur and ed < cur + len(word[i]) - 1))):
          flag = 1
          #print st, ed, cur, len(word[i]), word[i], tp
          break
        cur += len(word[i])
      if flag == 1: 
        cnt += 1 
      else: 
        tt += 1
        l.append(st)
        r.append(ed)
        t.append(tp)
    #print l, r, t
    l, r, t, num = getMaximumInterval(l, r, t)
    tt -= num
    cnt += num
    #print l, r, t
    for i in range(len(l)):
      cur = 1
      for j in range(len(word)):
        if cur >= l[i] and cur + len(word[j]) - 1 <= r[i]:
          tag[j] = ('B' if cur == l[i] else 'I') + '-' + t[i]
        #print cur, len(word[j]), word[j]
        cur += len(word[j])
    for i in range(len(word)):
      output.append(word[i] + ' ' + pos[i] + ' ' + tag[i] + '\n')
    output.append('\n')
    output_list.append(output)  

    #correction
    output = []
    sentence = doc.find('CORRECTION').text[1:-1]
    if (len(sentence.replace(' ', '')) != len(sentence)):
      tot += 1
      print (sentence)
    sentence = sentence.replace(' ', '')
    word = segmentor.segment(sentence.encode('utf-8'))  # 分词
    #word = []
    #for i in range(len(sentence)):
    #  word.append(sentence[i].encode('utf-8'))
    pos = postagger.postag(word)  # 词性标注
    tag = []
    
    word = list(word)
    pos = list(pos)
    for i in range(len(word)):
      word[i] = word[i].decode('utf-8')
      pos[i] = pos[i].decode('utf-8')

    word, pos = wordToChar(word, pos)
    for i in range(len(word)):
      tag.append('O')
    for i in range(len(word)):
      output.append(word[i] + ' ' + pos[i] + ' ' + tag[i] + '\n')
    output.append('\n')
    output_list.append(output)  

  print (cnt, tt)
  postagger.release()  # 释放模型
  segmentor.release()  # 释放模型
  print (len(output_list))
  random.seed(31415926)
  random.shuffle(output_list)
  file_object = open(train_file, 'w')
  num = 0
  for i in range(0, len(output_list) - test_size):
    ff = 0
    for j in range(len(output_list[i])):
      output_list[i][j].encode('utf-8')
      if (output_list[i][j] != '\n' and output_list[i][j].split(' ')[2][0] != 'O'): ff = 1
      file_object.write(output_list[i][j])
    if (ff == 0): num += 1
  print (num)
  file_object.close()
  file_object = open(test_file, 'w')
  lc = 0
  for i in range(len(output_list) - test_size, len(output_list)):
    ff = 0
    for j in range(len(output_list[i])):
      output_list[i][j].encode('utf-8')
      if (output_list[i][j] != '\n' and output_list[i][j].split(' ')[2][0] != 'O'): ff = 1
      file_object.write(output_list[i][j])
      lc += 1
    if (ff == 0): 
      num += 1
      #print lc
  print (num)
  file_object.close()
if __name__ == '__main__':
  main()
