#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet.defer import Deferred


def addBold(result):
    return '<b>%s</b>' % result


def addItalic(result):
    return '<i>%s</i>' % result


def printHTML(result):
    print result


def main():
    d = Deferred()
    d.addCallback(addBold)
    d.addCallback(addItalic)
    d.addCallback(printHTML)
    d.callback('mkkim')


def myCallback(result):
    print result


def myErrback(result):
    print result


def main2():
    d = Deferred()
    d.addCallbacks(myCallback, errback=myErrback)
    d.callback('mkkim')


if __name__ == '__main__':
    print 'Example 3-3 addMultiCallErrback.py'
    main2()


# EOF
