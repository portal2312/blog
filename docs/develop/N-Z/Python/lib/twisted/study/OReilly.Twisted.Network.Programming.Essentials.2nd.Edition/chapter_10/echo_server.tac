# -*- coding:utf8 -*-
from twisted.application import internet, service

from twisted.internet import protocol


class Echo(protocol.Protocol):
    def dataReceived(self, data):
        self.transport.write(data)


class EchoFactory(protocol.Factory):
    def buildProtocol(self, addr):
        return Echo()


print 'ex.6-3 echo_sever.tac'
application = service.Application('echo')
echoService = internet.TCPServer(8000, EchoFactory())
echoService.setServiceParent(application)
