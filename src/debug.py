from sklearn.metrics import classification_report
def eval(Y_test, Y_pred):
    #arget_names = ['O', 'B-R', 'I-R', 'B-M', 'I-M', 'B-S', 'I-S','B-W','I-W']
    print(classification_report(Y_test, Y_pred)) 
    
def get_data(train_data_path, flag):
    """
        input: 
            ---------------
            抱 B-v O
            怨 B-v O    -> sentence 1
            的 b-u O

            科 B-n O
            技 B-n O    -> sentence 2
            ---------------
    """
    data = open(train_data_path).readlines()
    training_data = []

    list1 = []
    list2 = []
    for i in range(len(data)):
        if flag == 'real':
            line_split = '\n'
        else:
            line_split = '\t\n'
        if data[i] != line_split:
            if flag == 'real':
                char, pos, err_true = data[i].split()
                list2.append(err_true)
            else:
                err_pred, char, pos, err_true = data[i].split()
                list2.append(err_pred)
            list1.append(char)
        else:
            
            training_data.append((list1, list2))
            list1 = []
            list2 = []
    return training_data

def color_tag(train_data):
    all_tag_nums = []
    sample_num = len(train_data)
    sample_num = 3000
    for i in range(sample_num):
        
        seq = train_data[i][0]
        tag = train_data[i][1]
        
        # Some replicate sample in test_CGED2016.txt
        seq_num = len(train_data[i][0])
        seq_half_before = train_data[i][0][:int(seq_num/2)]
        seq_half_end = train_data[i][0][int(seq_num/2):]
        if ''.join(seq_half_before) == ''.join(seq_half_end):
            tag_num = len(train_data[i][1])
            tag = train_data[i][1][:int(tag_num/2)]
            seq = train_data[i][0][:int(seq_num/2)]
        
        tag_nums = [0, 0, 0, 0] #R,M,S,W
        
        for j in range(len(tag)):
            err_type = tag[j].split('-')[-1]
            if err_type == 'R':
                seq[j] = "\033[1;31;40m"+seq[j]+"\033[0m"   #red
                tag_nums[0] += 1
            if err_type == 'M':
                seq[j] = "\033[1;30;44m"+seq[j]+"\033[0m"   #blue
                tag_nums[1] += 1
            if err_type == 'S':
                seq[j] = "\033[1;30;43m"+seq[j]+"\033[0m"   #yellow
                tag_nums[2] += 1
            if err_type == 'W':
                seq[j] = "\033[1;30;45m"+seq[j]+"\033[0m"   #magenta
                tag_nums[3] += 1
        all_tag_nums.append(tag_nums)
        print(i,''.join(seq))
    print('Done!')

def get_metric(real_data, pred_data):
    real_list = []
    pred_list = []
    sample_num = len(real_data)
    counter = 0
    for i in range(sample_num):
        real_seq = real_data[i][0]
        real_tag = real_data[i][1]
        pred_seq = pred_data[i][0]
        pred_tag = pred_data[i][1] 
        counter += len(real_seq)    
        # Some replicate sample in test_CGED2016.txt
        seq_num = len(real_data[i][0])
        seq_half_before = real_data[i][0][:int(seq_num/2)]
        seq_half_end = real_data[i][0][int(seq_num/2):]
        if ''.join(seq_half_before) == ''.join(seq_half_end):
            tag_num = len(real_data[i][1])
            real_tag = real_data[i][1][:int(tag_num/2)]
        
        # Debug
        if len(real_tag) != len(pred_tag):
            print("len of tag is inconsistent! ", i,len(real_tag), len(pred_tag), real_seq,counter,pred_seq)
        
        real_list.extend(real_tag)
        pred_list.extend(pred_tag)
    eval(real_list, pred_list)

if __name__ == '__main__':
    
    data_path_real = '../data/test_truth.txt'
    data_path_pred = 'merge.txt'
    real_data = get_data(data_path_real, 'real')
    pred_data = get_data(data_path_pred, 'pred')
    color_tag(real_data)
    #get_metric(real_data, pred_data)
