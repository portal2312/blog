#!/usr/bin/env python
# -*- coding:utf8 -*-
from twisted.internet import protocol, reactor


class QuoteProtocol(protocol.Protocol):
    def __init__(self, factory):
        self.factory = factory

    def connectionMade(self):
        self.sendQuote()

    def sendQuote(self):
        self.transport.write(self.factory.quote)

    def dataReceived(self, data):
        print '[Client][QuoteProtocol][dataReceived] ', data
        self.transport.loseConnection()


class QuoteClientFactory(protocol.ClientFactory):
    def __init__(self, quote):
        self.quote = quote

    def buildProtocol(self, addr):
        return QuoteProtocol(self)

    def clientConnectionFailed(self, connector, reason):
        print '[clientConnectionFailed] ', reason.getErrorMessage()
        maybeStopReactor()

    def clientConnectionLost(self, connector, reason):
        print '[clientConnectionLost] ', reason.getErrorMessage()
        maybeStopReactor()


def maybeStopReactor():
    global quote_counter
    quote_counter -= 1
    if not quote_counter:
        reactor.stop()


quotes = [
    'aaaaaaaa',
    'bbbbbbbb',
    'cccccccc',
]
quote_counter = len(quotes)


if __name__ == '__main__':
    print 'Example 2-4 quoteclient.py'
    while quotes:
        quote = quotes.pop(0)
        reactor.connectTCP('localhost', 8000, QuoteClientFactory(quote))
    reactor.run()


# EOF
