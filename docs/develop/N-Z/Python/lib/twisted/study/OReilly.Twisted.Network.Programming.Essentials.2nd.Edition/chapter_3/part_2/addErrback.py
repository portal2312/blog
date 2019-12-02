#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet.defer import Deferred
# from twisted.internet.protocol import Protocol, Factory


def myErrback(failure):
    print failure


def errCallback():
    d = Deferred()
    d.addErrback(myErrback)
    d.errback('myErrback')


if __name__ == '__main__':
    print 'Example 3-2 addErrback.py'
    errCallback()


# EOF
