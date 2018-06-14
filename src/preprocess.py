import os
import bz2
import logging
import shutil
from datetime import datetime
import json
import sys
from tqdm import tqdm
import time
import operator
import threading
import h5py
import numpy as np
import argparse
from copy import copy
import random

#local
import data


"""
Data_handler (base class)
"""
class Data_handler(object):

    def __init__(self,output_dir):

        self.output_dir = output_dir
        self.log_dir = os.path.join(self.output_dir,'log')

        #dirs & files
        date = datetime.now().strftime('%d-%m-%Y_%H:%M')
        self.log_file = os.path.join(self.log_dir,"log_%s.txt"%date)

    def set_logger(self):

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        format = '%(asctime)s\t[%(levelname)s]:\t%(message)s'
        logFormatter = logging.Formatter(format)
        logging.basicConfig(filename=self.log_file,
                            level=logging.DEBUG,
                            format=format)
        stream_hdl = logging.StreamHandler()
        stream_hdl.setFormatter(logFormatter)
        stream_hdl.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(stream_hdl)

    def logger_announce(self,str):
        logging.info('~*~*~*~*~*~~*~*~*~*~~*~*~*~')
        logging.info("DATA HANDLER - [%0s]"%str)
        logging.info('~*~*~*~*~*~~*~*~*~*~~*~*~*~')

    def merge_txt(self,source_list,target):

        with open(target, 'w') as outfile:
            for fname in source_list:
                with open(fname,'r') as infile:
                    for line in infile:
                        outfile.write(line)

    def merge_shuffle_txt(self,source_list,target):

        #open all source files
        f_list = [open(f_path,'r') for f_path in source_list]

        with open(target, 'w') as outfile:
            while len(f_list) > 0 :
                random_file = random.choice(f_list)
                print(random_file) #TODO - REMOVE
                random_line = random_file.readline()
                if random_line != '':
                    outfile.write(random_line)
                else:
                    random_file.close()
                    f_list.remove(random_file)

    def prepare_data(self):

        #create output folder
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        #set logger
        self.set_logger()

"""
Reddit_data_handler
"""
class Reddit_data_handler(Data_handler):
    """
    preprocessing reddit comments data
    # downloaded from:
    https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/
    # torrent magnet:
    magnet:?xt=urn:btih:7690f71ea949b868080401c749e878f98de34d3d&dn=reddit%5Fdata&tr=http%3A%2F%2Ftracker.pushshift.
    io%3A6969%2Fannounce&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80
    """
    def __init__(self,input_dir,output_dir,subreddit_list,debug_mode=False):

        super(Reddit_data_handler, self).__init__(output_dir)
        self.input_dir = input_dir
        self.subreddit_list = subreddit_list
        self.debug_mode = debug_mode

    def extract_bz2_file(self,file_path,save_path):
        try:
            with open(save_path, 'wb') as new_file, bz2.BZ2File(file_path, 'rb') as file:
                for data in iter(lambda: file.read(1024 * 1024), b''):
                    new_file.write(data)
        except:
            logging.info("UNEXPECTED ERROR DURING EXTRACT OF [%0s] - SKIPPING"%file_path)

    def text_block(self,files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    def json_process(self,input_path,output_dir):

        hist_path = os.path.join(output_dir,'hist.json')

        #try open statistics histogram
        if self.debug_mode:
            if os.path.exists(hist_path):
                with open(hist_path,'r') as f:
                    hist = json.load(f)
            else:
                hist = {}

        with open(input_path,'r') as in_json:
            line = 'START'
            count = 0

            #process single comment
            while line != '':
                line = in_json.readline()
                try:
                    comment = json.loads(line)
                    subreddit = comment['subreddit']
                    body = comment['body']

                    #add comment if not empty
                    if str(subreddit) in self.subreddit_list:
                        if body != '': #empty
                            if body[0] != '[' and not 'http' in body and not 'www' in body and len(body) > 3:
                                body = body.replace('\n',' ').replace('\r',' ').replace('\t',' ')\
                                    .replace('&lt;','<').replace('&gt;','>').replace('&amp;','&') #some modifications
                                with open(os.path.join(output_dir,subreddit + '.txt'),'a') as subreddit_file:
                                    subreddit_file.write(body + '\n')

                    #statistics
                    if self.debug_mode:
                        hist[subreddit] = hist.get(subreddit, 0) + 1

                except:
                    logging.info("[%0d][%s] is not a legal json - skipping"%(count,line))

                    #statistics
                    if self.debug_mode:
                        hist['ERROR'] = hist.get('ERROR', 0) + 1

                if self.debug_mode and count % 10000 == 0:
                    hist['TOTAL_COMMENTS'] = count
                    with open(hist_path, 'w') as f:
                        json.dump(hist,f)

                count += 1

    def summary(self,output_dir):

        hist_path = os.path.join(output_dir, 'hist.json')

        for file in os.listdir(output_dir):
            if file.endswith('.txt'):
                subreddit = file.split('.')[0]
                with open(os.path.join(output_dir,file), "r") as f:
                    comments = sum(bl.count("\n") for bl in self.text_block(f))
                logging.info("[%0s] contains [%0d] comments"%(subreddit,comments))

        # if self.debug_mode:
        #     with open(hist_path, 'r') as f:
        #         hist = json.load(f)
        #
        #     #sort
        #     hist_sort = sorted(hist.items(), key=operator.itemgetter(1))
        #     hist_sort = list(reversed(hist_sort))
        #
        #     top = 300
        #     logging.info("[[TOP %0d]]"%top)
        #     for i in range(top):
        #         logging.info(str(hist_sort[i]))

    def merge_subreddits(self):

        dirs = [os.path.join(self.output_dir,f)
                for f in os.listdir(self.output_dir)
                if os.path.isdir(os.path.join(self.output_dir,f)) and f != 'log']

        #merge
        for subreddit in self.subreddit_list:
            source = [os.path.join(dir, subreddit + '.txt')
                      for dir in dirs
                      if os.path.exists(os.path.join(dir, subreddit + '.txt'))]
            target = os.path.join(self.output_dir, subreddit + '.txt')
            self.merge_shuffle_txt(source,target)


        # #done - removing dirs
        # for dir in dirs:
        #     shutil.rmtree(dir)


    def process_single_file(self,bz2):

        tmp_json = bz2 + '.json'
        tmp_output = os.path.join(self.output_dir,os.path.basename(bz2).replace('.bz2',''))

        if os.path.exists(tmp_output):
            shutil.rmtree(tmp_output)
        os.mkdir(tmp_output)

        logging.info("extructing [%0s]..."%bz2)
        self.extract_bz2_file(bz2,tmp_json)
        logging.info("processing [%0s]..." % bz2)
        self.json_process(tmp_json,tmp_output)
        logging.info("deleting temp file [%0s]..." % bz2)
        os.remove(tmp_json)
        logging.info("[SUMMARY][%0s]"%bz2)
        self.summary(tmp_output)

    def prepare_data(self):

        super(Reddit_data_handler, self).prepare_data()

        self.logger_announce('start handling reddit data')

        #find all input files and prepare a list
        bz2_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if '.bz2' in file:
                    bz2_files.append(os.path.join(root,file))
        bz2_files.sort(key=lambda x: x.lower())

        # skip existing subreddits
        existing_subreddits = [s.split('.')[0] for s in os.listdir(self.output_dir) if s.endswith('.txt')]
        logging.info("%0s.txt already exists -> skipping" % existing_subreddits)
        self.subreddit_list = [e for e in self.subreddit_list if e not in existing_subreddits]

        # thread parameters
        num_bz2_files = len(bz2_files)
        threads = []
        threads_batch_size = 5
        thread_last_batch_size = num_bz2_files % threads_batch_size
        thread_batch_num = num_bz2_files // threads_batch_size

        for i in range(thread_batch_num):

            # prepare threads
            for j in range(threads_batch_size):
                thread = threading.Thread(target=self.process_single_file, args=(bz2_files[threads_batch_size * i + j],))
                threads.append(thread)

            # run threads
            for thread in threads:
                thread.start()

            # wait for done
            for thread in threads:
                thread.join()

            # remove threads
            threads = []

        # handle last batch
        for j in range(thread_last_batch_size):
            thread = threading.Thread(target=self.process_single_file, args=(bz2_files[threads_batch_size * thread_batch_num + j],))
            threads.append(thread)

        # run threads
        for thread in threads:
            thread.start()

        # wait for done
        for thread in threads:
            thread.join()

        # remove threads
        threads = []

        logging.info("[ALL DONE - MERGING]")
        self.merge_subreddits()

        self.logger_announce('done handling reddit data')

"""
H5_data_handler
"""

class H5_data_handler(Data_handler):

    def __init__(self,label2files_dict,output_dir,seq_len=128,debug_mode=False):

        super(H5_data_handler, self).__init__(output_dir)
        self.label2files_dict = label2files_dict
        self.debug_mode = debug_mode
        self.seq_len = seq_len

        #costs
        self.num_rows_per_file = 8 * 1024 * 1024
        self.num_labels = len(self.label2files_dict)
        self.num_rows_per_label = self.num_rows_per_file // self.num_labels
        self.num_rows_per_label_per_input_file = {label: self.num_rows_per_label // len(self.label2files_dict[label]) for label in self.label2files_dict.keys()}

        #label dicts
        self.label2num_dict = {label: i for i, label in enumerate(self.label2files_dict.keys())}
        self.num2label_dict = {i: label for i, label in enumerate(self.label2files_dict.keys())}


    def lines2tags(self,lines,label,seq_len=128):

        tags = np.ones([len(lines),seq_len],dtype=np.uint8) * data.END_TAG
        y = np.ones([len(lines),1],dtype=np.uint8) * self.label2num_dict[label]

        for i, line in enumerate(lines):
            for j in range(min(seq_len,len(lines[i]))):
                tags[i,j] = data.char2tag(lines[i][j])

        return tags , y

    def concat_and_shuffle(self,list_of_tuples):
        """
        :param list_of_tuples: list of tuples structed (tags,labels)
        :return: one tuple (tags,labels)
        """
        tags = tuple([tuple[0] for tuple in list_of_tuples])
        labels = tuple([tuple[1] for tuple in list_of_tuples])

        # concat
        if len(tags) > 1:
            tags = np.concatenate(tags,axis=0)
            labels = np.concatenate(labels, axis=0)
        else:
            tags = tags[0]
            labels = labels[0]

        # shuffle
        assert tags.shape[0] == labels.shape[0]
        permut = np.random.permutation(labels.shape[0])
        tags = tags[permut,:]
        labels = labels[permut, :]

        return tags, labels

    def save_h5(self,tags,labels,h5_path):
        with h5py.File(h5_path, 'w') as h5:
            h5_tags = h5.create_dataset('tags',shape=tags.shape,dtype=tags.dtype, compression="gzip")
            h5_tags[...] = tags
            h5_labels = h5.create_dataset('labels', shape=labels.shape, dtype=labels.dtype, compression="gzip")
            h5_labels[...] = labels

    def prepare_data(self):

        super(H5_data_handler, self).prepare_data()

        self.logger_announce('start handling h5 process')

        logging.info('[CREATING DATASET]')
        logging.info(str(self.label2files_dict))

        list_of_tuples = []

        for label in self.label2files_dict.keys():
            logging.info("[%0s] start processing label"%label)
            for file_path in self.label2files_dict[label]:
                logging.info("[%0s][%0s] start reading file" % (label, file_path))
                with open(file_path,'r') as file:
                    lines = []

                    for i in range(self.num_rows_per_label_per_input_file[label]):
                        line = file.readline()
                        if line != '':
                            lines.append(line)
                        else:
                            missing_lines = self.num_rows_per_label_per_input_file[label] - i
                            logging.info("missing %0d lines in label [%0s]"%(missing_lines,label))
                            break # for i in range(self.num_rows_per_label_per_input_file[label])

                logging.info("[%0s][%0s] start processing file" % (label, file_path))
                tags, y = self.lines2tags(lines,label,seq_len=self.seq_len)
                list_of_tuples.append((tags, y))

        logging.info("start compressing everything to H5")
        tags, y = self.concat_and_shuffle(list_of_tuples)
        self.save_h5(tags,y,h5_path=os.path.join(self.output_dir,'fff.h5')) #FIXME - TEMP

        self.h5_sanity_check(h5_path=os.path.join(self.output_dir,'fff.h5')) #FIXME - TEMP

        self.logger_announce('done handling h5 process')

    def h5_sanity_check(self,h5_path):

        logging.info("[%0s] start h5_sanity_check"%h5_path)

        with h5py.File(h5_path, 'r') as h5:

            for key in h5.keys():
                logging.info("[%0s] %0s"%(key,str(h5[key].shape)))

            logging.info("some samples:")

            for line in range(100,120):
                #extract sentance
                sent = ''
                i = 0
                while i < self.seq_len:
                    if h5['tags'][line,i] != data.END_TAG :
                        sent += data.tag2char(int(h5['tags'][line,i]))
                    else:
                        break
                    i += 1
                logging.info(self.num2label_dict[int(h5['labels'][line][0])] + '\t\t' + sent)

            labels = h5['labels']
            tags = h5['tags']

            # spent some to time to try and load all the data at once
            if self.debug_mode:
                t0 = time.time()
                labels = np.array(labels)
                tags = np.array(tags)
                logging.info("loading all data took [%0.1f SEC]"%(time.time() - t0))

            logging.info("data size in memory is [%0.2f MB]"%(tags.shape[0] * (tags.shape[1] + labels.shape[1]) / (1024. * 1024.)))
            logging.info('labels histogram:')
            unique, counts = np.unique(labels, return_counts=True)
            logging.info(str(dict(zip(unique, counts))))



def sandbox():
    pass

if __name__ == '__main__':
    desc = "DATA PREPROCESS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('mode', type=str, choices=['reddit_parse', 'reddit_h5', 'news_h5', 'news_h5_eng_only'],
                        help='supported modes are {reddit_parse,reddit_h5,news_h5,news_h5_eng_only}')
    FLAGS = parser.parse_args()

    if FLAGS.mode == 'reddit_parse':

        subreddits =     ['nfl','nba','gaming','soccer','movies','relationships','anime',
         'electronic_cigarette','Fitness','technology','pokemon','PokemonPlaza',
         'FIFA','Android','OkCupid','halo','bodybuilding','food','legaladvice',
         'skyrim','formula1','DnD','Guitar','Homebrewing','DIY','relationship_advice',
         'StarWars']

        R = Reddit_data_handler(input_dir='/Volumes/###/reddit-dataset/reddit_data',
                                output_dir='/Volumes/###/reddit-dataset/reddit_out',
                                subreddit_list=subreddits,
                                debug_mode=True)

        R.prepare_data()

        exit(0)

    elif FLAGS.mode == 'reddit_h5':
        pass

    elif FLAGS.mode == 'news_h5':

        label2files = {'en_news':
                       ['/Users/guytevet/Downloads/training-monolingual/news.2008.en.shuffled',
                        '/Users/guytevet/Downloads/training-monolingual/news.2009.en.shuffled',
                        '/Users/guytevet/Downloads/training-monolingual/news.2010.en.shuffled'],
                       'es_news':
                       ['/Users/guytevet/Downloads/training-monolingual/news.2008.es.shuffled'],
                       'de_news':
                       ['/Users/guytevet/Downloads/training-monolingual/news.2008.de.shuffled']
                       }

        out_dir = '/Users/guytevet/Downloads/training-monolingual/h5_tmp'

    elif FLAGS.mode == 'news_h5_en_only':
        pass
    else:
        raise NotImplementedError()

    H5 = H5_data_handler(label2files_dict=label2files,
                         output_dir=out_dir,
                         seq_len=64,
                         debug_mode=True)
    H5.prepare_data()




