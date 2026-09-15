from polypmeasure import PROJECT_ID, __version__
from polypmeasure.bootstrap import register


def test_project_identity():
    assert PROJECT_ID == "polypmeasure"
    assert __version__ == "0.1.0"


def test_bootstrap_is_lightweight_and_repeatable():
    assert register() == ()
    assert register() == ()
