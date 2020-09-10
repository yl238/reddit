import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datetime import datetime, timedelta
from app import app, db
from app.models import User, Post


def test_password_hashing():
    u = User(username='susan')
    u.set_password('cat')
    assert u.check_password('dog') == False
    assert u.check_password('cat') == True


@pytest.fixture(scope='function')
def setup_database():
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
    db.create_all()
    yield db
    db.session.remove()
    db.drop_all()


def test_dataset(setup_database):
    # create four users
    u1 = User(username='john', email='john@example.com')
    u2 = User(username='susan', email='susan@example.com')
    u3 = User(username='mary', email='mary@example.com')
    u4 = User(username='david', email='david@example.com')
    db.session.add_all([u1, u2, u3, u4])

    # create four posts
    now = datetime.utcnow()
    p1 = Post(body="post from john", author=u1,
                timestamp=now + timedelta(seconds=1))
    p2 = Post(body="post from susan", author=u2,
                timestamp=now + timedelta(seconds=4))
    p3 = Post(body="post from mary", author=u3,
                timestamp=now + timedelta(seconds=3))
    p4 = Post(body="post from david", author=u4,
                timestamp=now + timedelta(seconds=2))
    db.session.add_all([p1, p2, p3, p4])
    db.session.commit()

    # setup the followers
    u1.follow(u2)  # john follows susan
    u1.follow(u4)  # john follows david
    u2.follow(u3)  # susan follows mary
    u3.follow(u4)  # mary follows david
    db.session.commit()
    
    # check the followed posts of each user
    f1 = u1.followed_posts().all()
    f2 = u2.followed_posts().all()
    f3 = u3.followed_posts().all()
    f4 = u4.followed_posts().all()

    assert f1 == [p2, p4, p1]
    assert f2 == [p2, p3]
    assert f3 == [p3, p4]
    assert f4 == [p4]


def test_database(setup_database):
    u1 = User(username='john', email='john@examples.com')
    u2 = User(username='sue', email='sue@examples.com')
    db.session.add(u1)
    db.session.add(u2)
    db.session.commit()
    assert isinstance(u1.followed.all(), list)
    assert len(u1.followed.all()) == 0
    assert isinstance(u1.followers.all(), list)
    assert len(u1.followers.all()) == 0

    u1.follow(u2)
    db.session.commit()
    assert u1.is_following(u2) == True
    assert u1.followed.count() == 1
    assert u1.followed.first().username == 'sue'
    assert u2.followers.count() == 1
    assert u2.followers.first().username == 'john'

    u1.unfollow(u2)
    db.session.commit()
    assert u1.is_following(u2) == False
    assert u1.followed.count() == 0
    assert u2.followers.count() == 0


