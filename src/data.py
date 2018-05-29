##header here

import os
import random

#consts
# ASCII_OFFSET = 32
# ASCII_LEN = 126 - 32
# UNK_TAG = ASCII_LEN # '~'
# START_TAG = UNK_TAG + 1
# END_TAG = UNK_TAG + 2

space = [' ']
a_z = [chr(idx) for idx in range(ord('a'),ord('z')+1)]
A_Z = [chr(idx) for idx in range(ord('A'),ord('Z')+1)]
one_zero = [chr(idx) for idx in range(ord('1'),ord('0')+1)]
special = ['UNK', 'START' , 'END']
chars = space + a_z + A_Z + one_zero + special
char_dict = {chars[idx] : idx for idx in range(len(chars))}
tag_dict = {idx : chars[idx] for idx in range(len(chars))}

TAG_NUM = len(chars)
END_TAG = char_dict['END']

# def char2tag(char):
#     assert type(char) == str
#     assert len(char) == 1
#     tag = ord(char) - ASCII_OFFSET
#     if tag < 0 or tag > (ASCII_LEN - 1):
#         #print char #for debug UNKS
#         tag = UNK_TAG
#     return tag

def char2tag(char):
    assert type(char) == str
    assert len(char) == 1
    return char_dict.get(char,char_dict['UNK'])

# def tag2char(tag):
#     assert type(tag) == int
#     return chr(tag + ASCII_OFFSET)

def tag2char(tag):
    assert type(tag) == int
    return tag_dict[tag]

def text2sents(text):
    return text.split('\n')

def text2feed_tags(text, is_var_len=True, max_len=32, seq_len=32):
    sents = text2sents(text)
    feed_tags = []
    len_list = []
    for sent in sents:
        tags = [char_dict['START']] + ([char_dict['END']] * (seq_len + 1))
        if is_var_len == True:
            len = random.randrange(1,max_len+1)
        else:
            len = max_len
        for i, char in enumerate(sent[:len]):
            tags[i+1] = char2tag(char)
        feed_tags.append(tags)
        len_list.append(len)
    return feed_tags , len_list

def create_text_mask(len_list,seq_len=32,mode='th_extended'):
    mask_list = []
    for length in len_list:
        if mode == 'th_legacy':
            window_size = 1
            window_offset = length - 1
        elif mode == 'th_extended':
            window_size = random.randrange(1,length+1)
            window_offset = random.randrange(0,length-window_size+1)
        elif mode == 'full':
            window_size = length
            window_offset = 0
        else:
            raise TypeError('supported modes are {th_legacy,th_extended,full}')
        
        mask = [0] * (window_offset + 1) + [1] * (window_size) + [0] * (seq_len - window_size - window_offset + 1)
        assert len(mask) == seq_len + 2
        mask_list.append(mask)
    return mask_list

def load_sanity_data():
    sanity_text_path = os.path.join('..', 'data', 'europarl-v6.en')
    with open(sanity_text_path,'r', encoding='utf8') as file:
        text = file.read()
    return text[:10000000]

def create_shuffle_data(text,max_len,seq_len,mode):
    feed_tags , len_list = text2feed_tags(text,is_var_len=True,max_len=max_len,seq_len=seq_len)
    mask_list = create_text_mask(len_list,seq_len=seq_len,mode=mode)
    combined = list(zip(feed_tags, mask_list))
    random.shuffle(combined)
    feed_tags[:], len_list[:] = zip(*combined)
    return mask_list , feed_tags

def create_empty_data(sent_num,seq_len,max_len):


    # create dummy real sentence
    feed_tags = [[char_dict['START']] + ([char_dict['END']] * (seq_len + 1))] * sent_num

    mask_list = []
    for i in range(sent_num):

        #mask of the desired sentence length
        window_size = random.randrange(1, max_len + 1)
        window_offset = 0
        mask = [0] * (window_offset + 1) + [1] * (window_size) + [0] * (seq_len - window_size - window_offset + 1)
        assert len(mask) == seq_len + 2
        mask_list.append(mask)

    return mask_list , feed_tags

def test():

    sanity_text_path = os.path.join('..','data','europarl-v6.en')

    for char in 'hello world':
        tag = char2tag(char)
        print(str(tag) + '\t' + tag2char(tag))

    #chars = [tag2char(i) for i in range(ASCII_LEN + 1)]
    hist = [0] * (len(chars))


    with open(sanity_text_path,'r', encoding="utf8") as file:
        text = file.read()
        sents = text2sents(text)
        print('sanity text len: ' + str(len(text)))

    for sent in sents[:10000]:
        for char in sent:
            hist[char2tag(char)] += 1

    for i in range(len(chars)):
        print(chars[i] + '\t' + str(hist[i]))

    print('rare tags')
    for i in range(len(chars)):
        if hist[i] < 10:
            print(chars[i] + '\t' + str(hist[i]) + '\t' + str(char_dict[chars[i]]))

    # feed_tags , len_list = text2feed_tags(text[:100000],is_var_len=True,max_len=32)
    # mask_list = create_text_mask(len_list,max_len=32,mode='th_legacy')

    text = load_sanity_data()
    mask_list, feed_tags = create_shuffle_data(text,max_len=4,seq_len=32,mode='full')
    a=1


if __name__ == '__main__':
    test()

