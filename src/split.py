#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#seperate long sentences into short ones using delimeter "。！？"
def main():
  input_file = open(sys.argv[1], 'r')
  output_file = open(sys.argv[2], 'w')

  end = ['。', '！', '？']
  cnt = 0
  try:
    lines = input_file.readlines()
    for i in range(len(lines)):
      output_file.write(lines[i])
      if (lines[i] == '\n'): continue
      ch = lines[i].split(' ')[0]
      if (ch in end) and (lines[i + 1] != '\n'):
        output_file.write('\n')
        cnt += 1
  finally:
    input_file.close()
    output_file.close()
  print cnt

  


if __name__ == '__main__':
  main()
