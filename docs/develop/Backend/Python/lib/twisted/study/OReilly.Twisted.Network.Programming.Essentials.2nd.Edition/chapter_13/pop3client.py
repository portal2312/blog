# -*- coding:utf8 -*-
from twisted.mail import pop3client
from twisted.internet import reactor, protocol, defer
from cStringIO import StringIO
import email

USERNAME = 'recipient@localhost'
PASSWD = 'passwd'


class POP3LocalClient(pop3client.POP3Client):
    def serverGreeting(self, greeting):
        pop3client.POP3Client.serverGreeting(self, greeting)
        login = self.login(USERNAME, PASSWD).addCallbacks(
            self._loggedIn,
            self._ebLogin
        )

    def connectionLost(self, reason):
        print '>>> POP3LocalClient > connectionLost\n ', reason
        # XXX: FIXED
        # 예제는 소스상 작성되 있으나 자동으로
        # self.transport.lostConnection() 해서
        # 오히려 추가하면 오류 발생해서 주석 처리함
        # reactor.stop()
        pop3client.POP3Client.connectionLost(self, reason)

    def _loggedIn(self, result):
        return self.listSize().addCallback(self._getMessageSizes)

    def _ebLogin(self, result):
        print '>>> ebLogin', result
        self.transport.lostConnection()

    def _getMessageSizes(self, size):
        retrievers = []
        for i in range(len(size)):
            retrievers.append(self.retrieve(i).addCallback(
                self._gotMessageLines
            ))
        return defer.DeferredList(
            retrievers
        ).addCallback(
            self._finished
        )

    def _gotMessageLines(self, messageLines):
        for line in messageLines:
            print line

    def _finished(self, downloadResults):
        return self.quit()


class POP3ClientFactory(protocol.ClientFactory):
    def buildProtocol(self, addr):
        return POP3LocalClient()

    def clientConnectionLost(self, connector, reason):
        print '>>> POP3ClientFactory > clientConnectionLost\n', reason
        reactor.stop()


def run():
    reactor.connectTCP('localhost', 8000, POP3ClientFactory())
    reactor.run()


if __name__ == '__main__':
    print 'ex.13-7. pop3client.py'
    run()
