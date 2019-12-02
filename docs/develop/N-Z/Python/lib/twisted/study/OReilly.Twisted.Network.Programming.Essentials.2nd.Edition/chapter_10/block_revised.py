# -*- coding:utf8 -*-
import time

from twisted.internet import reactor, threads
from twisted.internet.task import LoopingCall


def blockingApiCall(arg):
    time.sleep(1)
    return arg


def nonblockingApiCall(arg):
    print arg


def printResult(result):
    print result


def finish(result):
    reactor.stop()


def run():
    d = threads.deferToThread(blockingApiCall, 'block')
    d.addCallback(printResult)
    d.addCallback(finish)

    LoopingCall(nonblockingApiCall, 'nonblock').start(.25)

    reactor.run()


if __name__ == '__main__':
    print 'ex.10-2 blocking_revised.py'
    run()
