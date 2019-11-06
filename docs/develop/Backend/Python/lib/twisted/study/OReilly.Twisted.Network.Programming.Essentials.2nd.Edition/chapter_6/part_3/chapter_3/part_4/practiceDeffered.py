#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet.defer import Deferred


def callback1(result):
    print 'callback1: %s' % result
    return result


def callback2(result):
    print 'callback2: %s' % result
    return result


def callback3(result):
    raise Exception('callback3')


def errback1(failure):
    print 'errback1: %s' % failure
    return failure


def errback2(result):
    raise Exception('errback2')


def errback3(result):
    print 'errback3: %s' % result
    return 'Everything is fine now.'


def bothback(result):
    result = '<end> %s' % result
    return result


def example1():
    d = Deferred()
    d.addCallback(callback1)
    d.addCallback(callback2)
    d.callback('mkkim')


def example2():
    d = Deferred()
    d.addCallback(callback1)
    d.addCallback(callback2)
    d.addCallback(callback3)
    d.callback('mkkim')


def example3():
    d = Deferred()
    d.addCallback(callback1)
    d.addCallback(callback2)
    d.addCallback(callback3)
    d.addErrback(errback3)
    d.callback('mkkim')


def example4():
    d = Deferred()
    d.addErrback(errback1)
    d.errback('mkkim')


def example5():
    d = Deferred()
    d.addErrback(errback1)
    d.addErrback(errback3)
    d.errback('mkkim')


def example6():
    d = Deferred()
    d.addErrback(errback2)
    d.errback('mkkim')


def example7():
    d = Deferred()
    d.addCallback(callback1)
    d.addCallback(callback2)
    d.addCallbacks(callback3, errback3)
    d.callback('mkkim')


def example8():
    '''
        callback3() > error >
        errback3() > ok >
        callback1() > result
    '''
    d = Deferred()
    d.addCallback(callback3)
    d.addCallbacks(callback2, errback3)
    d.addCallbacks(callback1, errback2)
    d.callback('mkkim')


def example9():
    d = Deferred()
    d.addCallback(callback3)
    d.addCallbacks(callback2, errback3)
    d.addCallbacks(callback1, errback2)
    d.addBoth(bothback)
    d.callback('mkkim')

if __name__ == '__main__':
    print 'Example 3-5 practiceDefferd.py'
    example8()


# EOF
