import os
import bz2
import logging
import shutil
from datetime import datetime
import json
import sys
from tqdm import tqdm
import operator

class Data_handler(object):

    def __init__(self,input_dir,output_dir):
        self.input_dir = input_dir
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

    def prepare_data(self):

        #create output folder
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        #set logger
        self.set_logger()

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

        super(Reddit_data_handler, self).__init__(input_dir,output_dir)
        self.subreddit_list = subreddit_list
        self.debug_mode = debug_mode

        #dirs & files
        self.tmp_json_file = os.path.join(self.output_dir,'tmp.json')
        self.histogram = os.path.join(self.output_dir,'histogram.json')


    def extract_bz2_file(self,file_path,save_path):
        with open(save_path, 'wb') as new_file, bz2.BZ2File(file_path, 'rb') as file:
            for data in tqdm(iter(lambda: file.read(1024 * 1024), b'')):
                new_file.write(data)

    def text_block(self,files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    def json_process(self,input_path):

        #try open statistics histogram
        if self.debug_mode:
            if os.path.exists(self.histogram):
                with open(self.histogram,'r') as f:
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
                    body = comment['body'].replace('\n',' ')

                    #add comment if not empty
                    if str(subreddit) in self.subreddit_list:
                        if body != '': #empty
                            if body[0] != '[' and not 'http' in body and not 'www' in body:
                                with open(os.path.join(self.output_dir,subreddit + '.txt'),'a') as subreddit_file:
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
                    with open(self.histogram, 'w') as f:
                        json.dump(hist,f)

                count += 1

    def summary(self):

        for file in os.listdir(self.output_dir):
            if file.endswith('.txt'):
                subreddit = file.split('.')[0]
                with open(os.path.join(self.output_dir,file), "r") as f:
                    comments = sum(bl.count("\n") for bl in self.text_block(f))
                logging.info("[%0s] contains [%0d] comments"%(subreddit,comments))

        if self.debug_mode:
            with open(self.histogram, 'r') as f:
                hist = json.load(f)

            #sort
            hist_sort = sorted(hist.items(), key=operator.itemgetter(1))
            hist_sort = list(reversed(hist_sort))

            top = 300
            logging.info("[[TOP %0d]]"%top)
            for i in range(top):
                logging.info(str(hist_sort[i]))

    def prepare_data(self):

        super(Reddit_data_handler, self).prepare_data()

        self.logger_announce('start handling reddit data')

        #find all input files and prepare a list
        bz2_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if '.bz2' in file:
                    bz2_files.append(os.path.join(root,file))

        # skip existing subreddits
        files = os.listdir(self.output_dir)
        existing_subreddits = [s.split('.')[0] for s in os.listdir(self.output_dir) if s.endswith('.txt')]
        logging.info("%0s.txt already exists -> skipping" % existing_subreddits)
        self.subreddit_list = [e for e in self.subreddit_list if e not in existing_subreddits]

        #process json
        for bz2 in bz2_files:
            if os.path.exists(self.tmp_json_file):
                os.remove(self.tmp_json_file)
            logging.info("extructing [%0s]..."%bz2)
            self.extract_bz2_file(bz2,self.tmp_json_file)
            logging.info("processing [%0s]..." % bz2)
            self.json_process(self.tmp_json_file)
            logging.info("deleting [%0s]..." % bz2)
            os.remove(self.tmp_json_file)
            logging.info("[SUMMARY]")
            self.summary()
        self.logger_announce('done handling reddit data')


def sandbox():
    pass

if __name__ == '__main__':
    # sandbox()
    R = Reddit_data_handler(input_dir='/Users/guytevet/nlp-final-project/datasets/reddit',
                            output_dir='/Users/guytevet/nlp-final-project/datasets/reddit_sandbox',
                            subreddit_list=['relationships','harrypotter'],
                            debug_mode=True)

    R.prepare_data()




