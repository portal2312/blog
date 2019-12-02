# -*- coding:utf8 -*-
from twisted.internet import protocol, reactor
from twisted.python import log


import sys


class Echo(protocol.Protocol):
    def dataReceived(self, data):
        log.msg(data)
        self.transport.write(data)


class EchoFactory(protocol.Factory):
    def buildProtocol(self, addr):
        return Echo()


def run():
    # XXX: Logging - files
    # log.startLogging(file=open('echo.log', 'w'))
    # XXX: Logging - direct print out
    log.startLogging(sys.stdout)

    reactor.listenTCP(8000, EchoFactory())
    reactor.run()


if __name__ == '__main__':
    print 'ex.7-1(7-2) logging_test.py'
    run()
