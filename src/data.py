##header here

import os
import random

#consts
ASCII_OFFSET = 32
ASCII_LEN = 126 - 32
UNK_TAG = ASCII_LEN # '~'
START_TAG = UNK_TAG + 1
END_TAG = UNK_TAG + 2

def char2tag(char):
    assert type(char) == str
    assert len(char) == 1
    tag = ord(char) - ASCII_OFFSET
    if tag < 0 or tag > (ASCII_LEN - 1):
        #print char #for debug UNKS
        tag = UNK_TAG
    return tag

def tag2char(tag):
    assert type(tag) == int
    return chr(tag + ASCII_OFFSET)

def text2sents(text):
    return text.split('\n')

def text2feed_tags(text, is_var_len=True, max_len=32):
    sents = text2sents(text)
    feed_tags = []
    len_list = []
    for sent in sents:
        tags = [START_TAG] + ([END_TAG] * (max_len + 1))
        if is_var_len == True:
            len = random.randrange(1,max_len+1)
        else:
            len = max_len
        for i, char in enumerate(sent[:len]):
            tags[i+1] = char2tag(char)
        feed_tags.append(tags)
        len_list.append(len)
    return feed_tags , len_list

def create_text_mask(len_list,max_len=32,mode='th_extended'):
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
        
        mask = [0] * (window_offset + 1) + [1] * (window_size) + [0] * (max_len - window_size - window_offset + 1)
        assert len(mask) == max_len + 2
        mask_list.append(mask)



    return mask_list


def test():

    sanity_text_path = os.path.join('..','data','europarl-v6.en')

    for char in 'hello world':
        tag = char2tag(char)
        print(str(tag) + '\t' + tag2char(tag))

    chars = [tag2char(i) for i in range(ASCII_LEN + 1)]
    hist = [0] * (ASCII_LEN + 1)


    with open(sanity_text_path,'r', encoding="utf8") as file:
        text = file.read()
        sents = text2sents(text)
        print('sanity text len: ' + str(len(text)))

    for sent in sents[:100000]:
        for char in sent:
            hist[char2tag(char)] += 1

    for i in range(len(chars)):
        print(chars[i] + '\t' + str(hist[i]))

    print('rare tags')
    for i in range(len(chars)):
        if hist[i] < 10:
            print(chars[i] + '\t' + str(hist[i]) + '\t' + str(ord(chars[i])))

    feed_tags , len_list = text2feed_tags(text[:100000],is_var_len=True,max_len=32)
    mask_list = create_text_mask(len_list,max_len=32,mode='full')
    a = 1




if __name__ == '__main__':
    test()

