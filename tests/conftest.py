from io import StringIO
import pandas as pd
import pytest


from reddit.config.base import config
from reddit.processing.data_management import load_dataset


@pytest.fixture(scope='session')
def raw_training_data():
    data = load_dataset(file_name=config.app_config.training_data_file) 
    return data

@pytest.fixture()
def uncleaned_dataset():
    TESTDATA = StringIO("""text\nYou are $3do2a\nAbced£\nhttps://gmail.com to be\nI've box""")
    df = pd.read_table(TESTDATA)
    return df

@pytest.fixture()
def mock_dataset():
    TESTDATA = StringIO("""title,body\nright_missing,\na1234,abcd\nhome 34,b 4\n,left_missing\nhurried back,done""")
    df = pd.read_table(TESTDATA, sep=',')
    return df

@pytest.fixture()
def pretokenized_dataset():
    TESTDATA=StringIO("""text\nhurried back\nI was finished\ndone""")
    df = pd.read_table(TESTDATA)
    return df

@pytest.fixture()
def predict_dataset():
    TESTDATA=StringIO(
        """title;score;num_comments;created_at;body;url
Is it common to have country especific tests?;0;1;2020-09-03 10:51:23;It is my first day on user testing and i couldnt take any due to me not living in one of the requested countrys. i would like to know how often that will happen;https://www.reddit.com/r/usertesting/comments/ilkfgh/is_it_common_to_have_country_especific_tests/
I booked a Live Conversation that didn’t ask me anything in the screener?;1;2;2020-09-03 05:48:19;All I was asked was “are you okay with this being recorded?” and then I qualified. I booked it but have no idea what it’s about or if i even qualify?;https://www.reddit.com/r/usertesting/comments/ilfa4e/i_booked_a_live_conversation_that_didnt_ask_me/
;1;2;2020-09-03 05:48:19;;https://www.reddit.com/r/usertesting/comments/ilfa4e/i_booked_a_live_conversation_that_didnt_ask_me/
Couldn't attend Live Conversation Test due to family emergency;7;8;2020-09-03 05:46:06;'Hi, has anyone been in a situation where you could not join a live conversation test due to a genuine reason?'""")
    df = pd.read_table(TESTDATA, sep=';')
    return df