# -*- coding:utf8 -*-
'''
ex.11-1 test_foo.py
'''
from twisted.trial import unittest


class MyFristTestCase(unittest.TestCase):
    '''
    함수까지 trial 하려면 현재 path 설정이 되어 있어야 한다.
    PYTHONPATH=$HOME/경로:$PYTHONPATH
    '''

    def test_something(self):
        self.assertTrue


# EOF
