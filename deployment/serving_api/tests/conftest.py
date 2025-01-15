import pytest  # type: ignore


@pytest.fixture(scope='module')
def sample_input():
    return {
        "posts": [
            {"text": "Why my answers not get any upvotes on Quora?"},
            {"text": "How do you train a pigeon to send messages?"},
        ]
    }
