# -*- coding:utf8 -*-
import time

from twisted.internet import reactor, threads
from twisted.internet.task import LoopingCall


def blockingApiCall(arg):
    time.sleep(1)
    return arg


def nonblockingCall(arg):
    print arg


def printResult(result):
    print result


def finish():
    reactor.stop()


def run():
    d = threads.deferToThread(blockingApiCall, 'Goose')
    d.addCallback(printResult)

    LoopingCall(nonblockingCall, 'Duck').start(.25)

    reactor.callLater(2, finish)
    reactor.run()


if __name__ == '__main__':
    print 'ex.10-1 blocking.py'
    run()
