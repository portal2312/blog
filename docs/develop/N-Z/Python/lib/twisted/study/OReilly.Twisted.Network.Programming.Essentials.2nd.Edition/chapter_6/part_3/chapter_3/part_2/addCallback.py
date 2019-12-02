#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet.defer import Deferred
# from twisted.internet.protocol import Protocol, Factory


def myCallback(result):
    print result


def addCallback():
    '''
    3.1
    '''
    print '3.1'
    d = Deferred()
    d.addCallback(myCallback)
    d.callback('myCallback')


if __name__ == '__main__':
    print 'Example 3-1 addCallback.py'
    addCallback()


# EOF
