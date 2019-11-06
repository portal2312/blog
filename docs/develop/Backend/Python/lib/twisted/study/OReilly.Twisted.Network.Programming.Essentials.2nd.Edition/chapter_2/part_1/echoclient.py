# -*- coding:utf-8 -*-
'''
ClientFactory
    buildProtocol
        Protocol
            dataReceived

reactor.connectTCP('localhost', 8000, Factory())
reactor.run()
'''

from twisted.internet import protocol, reactor


class EchoClient(protocol.Protocol):
    def connectionMade(self):
        self.transport.write('EchoClient - ConnectionMade')

    def dataReceived(self, data):
        print 'EchoClient: %r' % data
        self.transport.loseConnection()


class EchoFactory(protocol.ClientFactory):
    def buildProtocol(self, addr):
        return EchoClient()

    def clientConnectionLost(self, connector, reason):
        print 'Connection Lost.'
        reactor.stop()


if __name__ == '__main__':
    print 'Example 2-1 echoclient.py'

    reactor.connectTCP('127.0.0.1', 8000, EchoFactory())
    reactor.run()


# EOF
