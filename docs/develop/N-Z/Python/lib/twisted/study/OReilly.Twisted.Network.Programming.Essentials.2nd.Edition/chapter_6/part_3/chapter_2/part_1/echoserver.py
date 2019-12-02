# -*- coding:utf-8 -*-
'''
Factory
    buildProtocol
        Protocol
            dataReceived

reactor.listenTCP(port, Factory())
reactor.run()
'''
from twisted.internet import protocol, reactor


class Echo(protocol.Protocol):
    def dataReceived(self, data):
        self.transport.write(data)


class EchoFactory(protocol.Factory):
    def buildProtocol(self, addr):
        return Echo()


if __name__ == '__main__':
    print 'Example 2-1 echoserver.py'

    reactor.listenTCP(8000, EchoFactory())
    reactor.run()


# EOF
