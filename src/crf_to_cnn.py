import sys

#Tranform the data format.
def main():
  crf_file = open(sys.argv[1], 'r')
  cnn_file = open(sys.argv[2], 'w')  
  try:
    crf_lines = crf_file.readlines()
    cnn_line =  ""
    cnt = 0
    for line in crf_lines:
      if line == '\n':
        cnn_file.write(cnn_line[0:-1] + '\n')
        cnn_line = ""
        cnt += 1
        continue
      cnn_line += line[0:-1].replace(' ', '&') + ' '
  finally:
    crf_file.close()
    cnn_file.close()
    print cnt
  
if __name__ == '__main__':
  main()
