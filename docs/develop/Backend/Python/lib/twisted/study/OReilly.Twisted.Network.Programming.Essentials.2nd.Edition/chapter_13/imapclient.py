# -*- coding:utf8 -*-
import sys
from twisted.internet import protocol, reactor
from twisted.mail import imap4
from twisted.python import log

USERNAME = 'recipient@localhost'
PASSWD = 'passwd'


class IMAP4LocalClient(imap4.IMAP4Client):
    def connectionMade(self):
        log.msg('Login: %s/%s' % (USERNAME, PASSWD))
        self.login(USERNAME, PASSWD).addCallbacks(
            self._getMessages,
            self._ebLogin
        )

    def connectionLost(self, reason):
        log.msg(reason)
        reactor.stop()

    def _ebLogin(self, result):
        log.msg(result)
        self.transport.loseConnection()

    def _getMessages(self, result):
        log.msg('getMessages: %r' % (result,))
        return self.list('', '*').addCallback(self._cbPickMailbox)

    def _cbPickMailbox(self, result):
        mbox = filter(lambda x: 'Inbox' in x[2], result)[0][2]
        return self.select(mbox).addCallback(self._cbExamineMbox)

    def _cbExamineMbox(self, result):
        return self.fetchMessage('1:*', uid=False).addCallback(
            self._cbFetchMessages
        )

    def _cbFetchMessages(self, result):
        for seq, message in result.iteritems():
            print seq, message['RFC822']


class IMAP4ClientFactory(protocol.ClientFactory):
    def buildProtocol(self, addr):
        return IMAP4LocalClient()

    def clientConnectionFailed(self, connector, reason):
        log.msg(reason)
        reactor.stop()


def run():
    log.startLogging(sys.stdout)
    reactor.connectTCP('localhost', 8000, IMAP4ClientFactory())
    reactor.run()


if __name__ == '__main__':
    print 'ex.13-5 imapclient.py'
    run()


# EOF
