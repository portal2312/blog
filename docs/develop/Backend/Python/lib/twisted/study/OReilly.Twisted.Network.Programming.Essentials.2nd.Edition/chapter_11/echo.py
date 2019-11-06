# -*- coding:utf8 -*-
'''
ex.11-2 echo.py
'''
from twisted.internet import protocol


class Echo(protocol.Protocol):
    def dataReceived(self, data):
        self.transport.write(data)


class EchoFactory(protocol.Factory):
    def buildProtocol(self, addr):
        return Echo()


# EOF
