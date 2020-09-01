import praw
import pandas as pd 
from datetime import datetime

def get_date(created):
    return datetime.fromtimestamp(created)

class Post(object):
    def __init__(self, session):
        self.title = session.title
        self.score = session.score
        self.url = session.url
        self.num_comments = session.num_comments
        self.created_at = get_date(session.created)
        self.body = session.selftext

    def __dict__(self):
        return {'title': self.title, 'score': self.score, 'num_comments': self.num_comments,
                'created_at': self.created_at, 'body': self.body, 'url': self.url}

class RedditScraper(object):
    def __init__(self, subreddit):
        self.reddit = praw.Reddit('UserTesting')
        self.subreddit = self.reddit.subreddit(subreddit)
    
    def scrape(self, limit):
        new_posts = self.subreddit.new(limit=limit)
        for session in new_posts:
            yield Post(session)

    def get_data(self, limit):
        data = []
        for post in self.scrape(limit):
            data.append(post.__dict__())
        df = pd.DataFrame.from_records(data)
        return df