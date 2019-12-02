#!/usr/bin/env python
# -*- coding:utf8 -*-
'''
<server>
python chatserver.py

<client>
telnet localhost 8000

'''
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver
from twisted.internet import reactor


class ChartProtocol(LineReceiver):
    def __init__(self, factory):
        self.factory = factory
        self.name = None
        self.state = 'Register'
        self.users = None

    def connectionMade(self):
        self.sendLine('Login: ')

    def connectionLost(self, reason):
        if self.name not in self.factory.users:
            return

        del self.factory.users[self.name]
        self.users = self.factory.users.keys()
        self.broadcastMessage(
            '%s has left the channel.\nusers: %r' % (
                self.name,
                self.users,
            )
        )

    def lineReceived(self, line):
        if self.state == 'Register':
            self.handle_REGISTER(line)
        else:
            self.handle_CHAT(line)

    def handle_REGISTER(self, name):
        if name in self.factory.users:
            self.sendLine('Name taken, please choose another.')
            return

        self.name = name
        self.factory.users[name] = self
        self.users = self.factory.users.keys()
        self.sendLine('Hello, %s\nusers: %r' % (name, self.users))
        self.broadcastMessage(
            '%s has joined the channel.\nusers: %r' % (
                name,
                self.users
            )
        )
        self.state = 'CHAT'

    def handle_CHAT(self, msg):
        msg = '<%s> %s' % (self.name, msg)
        self.broadcastMessage(msg)

    def broadcastMessage(self, msg):
        for name, protocol in self.factory.users.iteritems():
            if protocol != self:
                protocol.sendLine(msg)


class ChartFactory(Factory):
    def __init__(self):
        self.users = {}

    def buildProtocol(self, addr):
        return ChartProtocol(self)


if __name__ == '__main__':
    print 'Example 2-5 chatserver.py'
    reactor.listenTCP(8000, ChartFactory())
    reactor.run()


# EOF
