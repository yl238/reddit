import praw
import pandas as pd
import logging


if __name__ == '__main__':
    
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    logger = logging.getLogger('prawcore')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    reddit = praw.Reddit('UserTesting')

    submission = reddit.submission(url="https://www.reddit.com/r/usertesting/comments/eavmei/no_tests_available_do_they_really_only_come_once/")

    submission.comments.replace_more(limit=None)
    


    