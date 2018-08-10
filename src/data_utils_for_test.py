import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from pyltp import Segmentor
from pyltp import Postagger

def wordToChar(word, pos):
  resw = []
  resp = []
  for i in range(len(word)):
    for j in range(len(word[i])):
      resw.append(word[i][j])
      resp.append(('B-' if j == 0 else 'I-') + pos[i])
  return resw, resp


def main():
  segmentor = Segmentor()
  segmentor.load('./cws.model')
  postagger = Postagger()
  postagger.load('./pos.model')
  file_object = open(sys.argv[1], 'r')
  sid = []
  output_list = []
  try:
    all_lines = file_object.readlines()
    lc = 0
    tot = 0
    for line in all_lines:
      output = []
      lc += 1
      item = line.split('\t')
      sid.append(item[0])
      sentence = item[1][0:-1]
      #print sentence.decode('utf-8')
      if (len(sentence.replace(' ', '')) != len(sentence)):
        tot += 1
        print lc
      sentence = sentence.replace(' ', '')
      word = segmentor.segment(sentence.encode('utf-8'))
      pos = postagger.postag(word)
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
    print tot
  finally:
    file_object.close()

  file_object = open(sys.argv[2], 'w')
  negative_num = 0
  for i in range(len(output_list)):
    ff = 0
    for j in range(len(output_list[i])):
      output_list[i][j].encode('utf-8')
      if (output_list[i][j] != '\n' and output_list[i][j].split(' ')[2][0] != 'O'): ff = 1
      file_object.write(output_list[i][j])
    if (ff == 0): negative_num += 1
  print negative_num

  file_object = open('SID.txt', 'w')
  for i in range(len(sid)):
    file_object.write(sid[i] + '\n')
  file_object.close()
    


if __name__ == '__main__':
  main()