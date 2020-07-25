import fnmatch
import os
import re

pre_in_folder = 'sample_data/input/pre'
post_in_folder = 'sample_data/input/post'
output_folder = 'sample_data/output'


def main():
    pre_files = get_files(pre_in_folder)
    post_files = get_files(post_in_folder)
    file_valid_check(pre_files, post_files)

def get_files(where, which='*.png'):

    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    return [name for name in os.listdir(where) if rule.match(name)]


def file_valid_check(pre, post):
    if len(pre) != len(post):
        return False



if __name__ == '__main__':
    main()
