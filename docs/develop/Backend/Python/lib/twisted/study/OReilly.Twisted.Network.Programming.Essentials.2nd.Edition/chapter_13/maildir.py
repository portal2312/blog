# -*- coding:utf8 -*-
import os
import sys

from email.header import Header
from zope.interface import implements

from twisted.internet import reactor
from twisted.mail import smtp, maildir
from twisted.python import log


class LocalMessageDelivery(object):
    implements(smtp.IMessageDelivery)

    def __init__(self, protocol, baseDir):
        self.protocol = protocol
        self.baseDir = baseDir

    def receivedHeader(self, helo, origin, recipients):
        clientHostname, clientIP = helo
        myHostname = self.protocol.transport.getHost().host
        headerValue = 'from %s by %s with ESMTP ; %s' % (
            clientHostname, myHostname, smtp.rfc822date
        )
        return 'Received: %s' % Header(headerValue)

    def validateFrom(self, helo, origin):
        return origin

    def _getAddressDir(self, address):
        return os.path.join(self.baseDir, '%s' % address)

    def validateTo(self, user):
        if user.dest.domain == 'localhost':
            return lambda: MaildirMessage(
                self._getAddressDir(str(user.dest))
            )
        else:
            log.msg('Received email for invalid recipient %s' % user)
            raise smtp.SMTPBadRcpt(user)


class MaildirMessage(object):
    implements(smtp.IMessage)

    def __init__(self, userDir):
        if not os.path.exists(userDir):
            os.mkdir(userDir)
        inboxDir = os.path.join(userDir, 'Inbox')
        self.mailbox = maildir.MaildirMailbox(inboxDir)
        self.lines = []

    def lineReceived(self, line):
        self.lines.append(line)

    def eomReceived(self):
        print 'New message received.'
        self.lines.append('')
        messageData = '\n'.join(self.lines)
        return self.mailbox.appendMessage(messageData)

    def connectionLost(self):
        print 'Connection lost unexpectedly!'
        # 예기치 않게 연결을 잃어버렸으면, contents 를 저장하지 않는다.
        del(self.liens)


class LocalSMTPFactory(smtp.SMTPFactory):
    def __init__(self, baseDir):
        self.baseDir = baseDir

    def buildProtocol(self, addr):
        proto = smtp.ESMTP()
        proto.delivery = LocalMessageDelivery(proto, self.baseDir)
        return proto


def run():
    log.startLogging(sys.stdout)
    reactor.listenTCP(
        8001,
        LocalSMTPFactory('/home/mkkim/twisted_example/chapter_13/mail')
    )
    reactor.run()


if __name__ == '__main__':
    print 'ex.13-3 smtp_server.py'
    run()
