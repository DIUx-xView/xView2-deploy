import fnmatch
import os
import re

PRE_DIR = 'sample_data/input/pre'
POST_DIR = 'sample_data/input/post'
OUTPUT_DIR = 'sample_data/output'


def main():

    pre_files = get_files(PRE_DIR)
    post_files = get_files(POST_DIR)
    file_valid_check(pre_files, post_files)


def get_files(where, which='*.png'):

    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    return [name for name in os.listdir(where) if rule.match(name)]


def file_valid_check(pre, post):
    if len(pre) != len(post):
        return False



if __name__ == '__main__':
    main()
