#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#recover testing output of original length
def main():
  tag_file = open(sys.argv[1], 'r')
  test_file = open(sys.argv[2], 'r') 
  output_file = open(sys.argv[3], 'w')

  try:
    tag_lines = tag_file.readlines()
    test_lines = test_file.readlines()
    # get seq len of test data
    l = []
    cnt = 0
    
    seq = []
    seqs = []
    for i in range(len(test_lines)):
      if len(test_lines[i]) <= 2:
        if cnt == 0: print '!!!'
        l.append(cnt)
        cnt = 0
        continue
      cnt += 1
    
    j = 0
    cnt = 0
    t = []
    for i in range(len(tag_lines)):
      if tag_lines[i] == '\n':
        continue
      cnt += 1
      t.append(tag_lines[i])
      if cnt == l[j]:
        for x in t:
          output_file.write(x)
        output_file.write('\n')
        j += 1
        cnt = 0
        t = []
  finally:
    tag_file.close()
    test_file.close()
    output_file.close()


if __name__ == '__main__':
  main()
