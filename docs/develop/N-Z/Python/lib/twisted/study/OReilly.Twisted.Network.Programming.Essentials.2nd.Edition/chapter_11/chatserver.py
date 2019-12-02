# -*- coding:utf8 -*-
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver


class ChatProtocol(LineReceiver):
    def __init__(self, factory):
        self.factory = factory
        self.name = None
        self.state = 'REGISTER'

    def connectionMade(self):
        self.sendLine('Login-name: ')

    def connectionLost(self, reason):
        if self.name not in self.factory.users:
            return

        del self.factory.users[self.name]

        self.broadcastMessage(
            '%s has left the channel. (%s)' % (
                self.name,
                self.factory.users.keys()
            )
        )

    def lineReceived(self, line):
        if self.state == 'REGISTER':
            self.handle_REGISTER(line)
        elif self.state == 'CHAT':
            self.handle_CHAT(line)

    def handle_REGISTER(self, name):
        if name in self.factory.users:
            self.sendLine('name taken, please choose another.')
            return
        self.sendLine('Login success. (%s)' % (name, ))
        self.broadcastMessage('"%s" has joined the channel.' % (name, ))
        self.name = name
        self.factory.users[name] = self
        self.state = 'CHAT'

    def handle_CHAT(self, msg):
        msg = '[%s] %s' % (self.name, msg)
        self.broadcastMessage(msg)

    def broadcastMessage(self, msg):
        for name, protocol in self.factory.users.iteritems():
            if protocol != self:
                protocol.sendLine(msg)


class ChatFactory(Factory):
    def __init__(self):
        self.users = {}

    def buildProtocol(self, addr):
        return ChatProtocol(self)
