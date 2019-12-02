#!/usr/bin/env python
# -*- coding:utf8 -*-
import sys

from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.protocol import Protocol
from twisted.web.client import Agent


class ResourcePrinter(Protocol):
    def __init__(self, finished):
        self.finished = finished

    def dataReceived(self, data):
        print data

    def connectionLost(self, reason):
        self.finished.callback(None)


def printResource(response):
    finished = Deferred()
    response.deliverBody(ResourcePrinter(finished))
    return finished


def printError(failure):
    return


def stop(result):
    reactor.stop()


def run():
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: python agent_print_resource.py <URL>'
        exit(1)

    agent = Agent(reactor)
    d = agent.request('GET', sys.argv[1])
    d.addCallbacks(printResource, printError)
    d.addBoth(stop)

    reactor.run()


if __name__ == '__main__':
    print 'ex.5-3 agent_print_resource.py'
    run()
