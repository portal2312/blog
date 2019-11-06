#!/usr/bin/env python
# -*- coding:utf8 -*-
from twisted.internet.protocol import Factory
from twisted.internet import protocol, reactor


class QuoteProtocol(protocol.Protocol):
    # direct - buildProtocol and self.Factory
    # def __init__(self, factory):
    #     self.factory = factory

    def connectionMade(self):
        print 'connectionMade'
        self.factory.numConnections += 1

    def dataReceived(self, data):
        print '[Server][QuoteProtocol][dataReceived] %r, %r, %r' % (
            self.factory.numConnections,
            data,
            self.getQuote(),
        )
        self.transport.write(self.getQuote())
        self.updateQuote(data)

    def connectionLost(self, reason):
        self.factory.numConnections -= 1

    def getQuote(self):
        return self.factory.quote

    def updateQuote(self, quote):
        self.factory.quote = quote


class QuoteFactory(Factory):
    numConnections = 0

    # direct - buildProtocol and self.Factory
    protocol = QuoteProtocol

    def __init__(self, quote=None):
        self.quote = quote or 'None'

    # static - buildProtocol and Factory
    # def buildProtocol(self, addr):
    #     return QuoteProtocol(self)


if __name__ == '__main__':
    print 'Example 2-3 quoteserver.py'

    reactor.listenTCP(8000, QuoteFactory())
    reactor.run()


# EOF
