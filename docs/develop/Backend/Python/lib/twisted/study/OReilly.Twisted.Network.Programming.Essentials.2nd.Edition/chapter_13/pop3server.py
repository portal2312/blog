# -*- coding:utf8 -*-
import os
import sys
from zope.interface import implements

from twisted.cred import checkers, portal
from twisted.internet import protocol, reactor
from twisted.mail import maildir, pop3
from twisted.python import log


class UserInbox(maildir.MaildirMailbox):
    def __init__(self, userDir):
        inboxDir = os.path.join(userDir, 'Inbox')
        # XXX:
        # 이미 maildir.MaildirMailbox는 inheritance 했으나
        # maildir.MaildirMailbox.__init__ 다시 사용하는 이유는
        # __init__ 안의 정의된 내용을 사용하기 위해 (정의 내용은 definition 확인)
        # class, def namespace 유지하면서 userDir 인자 값 완성후 전달 함
        maildir.MaildirMailbox.__init__(self, inboxDir)


class MailUserRealm(object):
    implements(portal.IRealm)

    def __init__(self, baseDir):
        self.baseDir = baseDir

    def requestAvatar(self, avatarId, *interfaces):
        if pop3.IMailbox not in interfaces:
            raise NotImplementedError(
                'This realm only supports the pop3.IMailbox interface.'
            )

        userDir = os.path.join(self.baseDir, avatarId)
        avatar = UserInbox(userDir)
        return pop3.IMailbox, avatar, lambda: None


class POP3ServerProtocol(pop3.POP3):
    def lineReceived(self, line):
        print '>>> Client: ', line
        pop3.POP3.lineReceived(self, line)

    def sendLine(self, line):
        print '>>> Server: ', line
        pop3.POP3.sendLine(self, line)


class POP3Factory(protocol.Factory):
    def __init__(self, portal):
        self.portal = portal

    def buildProtocol(self, addr):
        proto = POP3ServerProtocol()
        proto.portal = self.portal
        return proto


def run():
    log.startLogging(sys.stdout)

    dataDir = sys.argv[1]
    if dataDir:
        dataDir = os.path.abspath(dataDir)

    portal_ = portal.Portal(MailUserRealm(dataDir))
    portal_.registerChecker(
        checkers.FilePasswordDB(os.path.join(dataDir, 'passwds.txt'))
    )

    reactor.listenTCP(8000, POP3Factory(portal_))
    reactor.run()


if __name__ == '__main__':
    print 'ex.13-6. pop3server.py'
    run()
