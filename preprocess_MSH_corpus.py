import arff
import os
import html
import re
import random
import sys

class Dataset:

    def __init__(self, corpus_path = None):
        self.corpus = []
        self.index = dict()
        if corpus_path is not None:
            self.load(corpus_path)

    """ preprocessing a single title+abstract text
    """
    def preprocess_text(self, text):
        text = html.unescape(text)
        text = re.sub(r'<e>(.*)</e>', r'\1', text) # remove marker for acronym that is to be disambiguated
        #text = re.sub(r'\.(\S)', r'. \1', text) # separating space between titel and abstract text
        text = re.sub(r'\W+(\s)', r' \1', text) # remove all non-word characters (brackets, dashes, dots,...) at the beginning of words
        text = re.sub(r'(\s)\W+', r' \1', text) # remove all non-word characters (brackets, dashes, dots,...) at the end of words
        return text.lower()

    """ loads all files with name "<acronym>_(.*).arff" from the provided corpus_path.
    extracts the acronym from the file path and title-abstract text as well as correct classification label for each sample from the file contents.
    applies preprocess_text to each sample text and stores them in the self.corpus collection (as well as updating self.index)
    """
    def load(self, corpus_path):
        self.data = []
        self.acronym_map = dict()

        file_list = set(os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if filename.endswith(".arff"))
        not_found_acronyms = []
        id = 0
        for filename in file_list:
            acronym = os.path.basename(filename).split("_")[0]
            filepath = os.path.join(corpus_path, filename)
            with open(filepath, "r") as f:
                arff_contents = arff.load(f)
                for item in arff_contents['data']:
                    sense_label = item[2]
                    text = self.preprocess_text(item[1])
                    self.corpus.append((acronym, sense_label, text))
                    try:
                        meaning_map = self.index[acronym]
                    except KeyError:
                        meaning_map = dict()
                        self.index[acronym] = meaning_map
                    try:
                        meaning_map[sense_label] = meaning_map[sense_label] + [id]
                    except KeyError:
                        meaning_map[sense_label] = [id]
                    id = id + 1

    """ returns all loaded acronyms
    """
    def get_acronyms(self):
        return sorted(acronym for acronym in self.index.keys())


    """ returns statistics about the loaded acronyms
    the result is a 2-tuple:
        the first components is a dictionary acronym => sense_label => sample count
        the second component is a histogram of number of senses/meanings per acronym (with bins 1 - 20)
    """
    def get_acronym_stats(self):
        acronym_stats = dict()
        sense_count_bins = [0] * 20
        for acronym in self.index:
            sense_count = len(self.index[acronym])
            sense_count_bins[sense_count - 1] += 1
            sample_counts = dict()
            for sense in self.index[acronym]:
                sample_counts[sense] = len(self.index[acronym][sense])
            acronym_stats[acronym] = sample_counts
        return acronym_stats, sense_count_bins

    """ concatenates the texts of the indicated samples into one line and dumps them into the file specified by filename.
    argument sample_ids allows to specify which sample texts will be dumped. if left None, all samples will be written.
    """
    def dump_single_line_texts_file(self, sample_ids = None, filename='corpus_out.txt'):
        if sample_ids is None:
            sample_ids = range(len(self.corpus))
        with open(filename, 'w') as f:
            f.write(' '.join(self.corpus[sample_id][2] for sample_id in sample_ids))
            f.flush()

    """ writes all indicated samples as seperate lines into the specified file. each line has the format <acronym>;<sense label>;<text>.
    argument sample_ids allows to specify which sample texts will be dumped. if left None, all samples will be written.
    """
    def dump_multi_line_corpus_file(self, sample_ids = None, filename='corpus_tagged_out.txt'):
        if sample_ids is None:
            sample_ids = range(len(self.corpus))
        with open(filename, 'w') as f:
            for sample_id in sample_ids:
                f.write(';'.join(self.corpus[sample_id]) + '\n')
            f.flush()

    """ performs a train/test split and returns train_set_sample_ids, test_set_sample_ids.
    performs a train/test split with ratio (1-test_ratio):test_ratio for each acronym-sense sample subset. joins all subset splits and shuffles before returning.
    """
    def get_train_test_split_ids(self, test_ratio = 0.2):
        train_set, test_set = [], []
        for acronym in self.index:
            for sense in self.index[acronym]:
                sense_sample_set = self.index[acronym][sense]
                sense_count = len(sense_sample_set)
                sense_sample_set = random.sample(sense_sample_set, sense_count) # imitiate random.shuffle but have a return value instead of manipulating list in pace
                                                                                # (which would be bad as the list is a reference into our object's main index)
                test_set = test_set + sense_sample_set[:int(sense_count * test_ratio)]
                train_set = train_set + sense_sample_set[int(sense_count * test_ratio):]
        random.shuffle(train_set)
        random.shuffle(test_set)
        return train_set, test_set

    """ performs a train/test split and returns train_set_sample_ids, test_set_sample_ids for a single acronym.
    performs a train/test split with ratio (1-test_ratio):test_ratio for each acronym-sense sample subset. joins all subset splits and shuffles before returning.
    """
    def get_train_test_split_ids_for_acronym(self, acronym, test_ratio = 0.2):
        train_set, test_set = [], []
        for sense in self.index[acronym]:
            sense_sample_set = self.index[acronym][sense]
            sense_count = len(sense_sample_set)
            sense_sample_set = random.sample(sense_sample_set, sense_count) # imitiate random.shuffle but have a return value instead of manipulating list in pace
                                                                            # (which would be bad as the list is a reference into our object's main index)
            test_set = test_set + sense_sample_set[:int(sense_count * test_ratio)]
            train_set = train_set + sense_sample_set[int(sense_count * test_ratio):]
        random.shuffle(train_set)
        random.shuffle(test_set)
        return train_set, test_set


if __name__ == '__main__':
    dataset = Dataset(sys.argv[1])
    print(dataset.get_acronym_stats())
    train_set, test_set = dataset.get_train_test_split_ids()
    dataset.dump_single_line_texts_file(train_set, "data/msh_train.txt")
    dataset.dump_multi_line_corpus_file(train_set, "data/msh_train_with_labels.txt")
    dataset.dump_multi_line_corpus_file(test_set, "data/msh_test_with_labels.txt")
