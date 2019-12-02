# -*- coding:utf8 -*-
import sys

from email.header import Header
from zope.interface import implements

from twisted.internet import defer, reactor
from twisted.mail import smtp
from twisted.python import log


class StdoutMessageDelivery(object):
    # XXX: IMessageDelivery
    #   (= from twisted.mail.interfaces import IMessageDelivery)
    # mail 전달 역할
    #   header 가공,
    #   To, From 에 대한 유효성 검사
    implements(smtp.IMessageDelivery)

    def __init__(self, protocol):
        self.protocol = protocol

    def receivedHeader(self, helo, origin, recipients):
        clientHostname, _ = helo
        myHostname = self.protocol.transport.getHost().host
        headerValue = 'from %s by %s with ESMTP ; %s' % (
            clientHostname, myHostname, smtp.rfc822date
        )
        return 'Received: %s' % Header(headerValue)

    def validateFrom(self, helo, origin):
        # XXX: Accepted all sender
        return origin

    def validateTo(self, user):
        # XXX: @localhost 가 수령인인 메일만 수락한다.
        if user.dest.domain == 'localhost':
            return StdoutMessage
        else:
            log.msg('Received email for invalid recipient %s' % user)
            raise smtp.SMTPBadRcpt(user)


class StdoutMessage(object):
    # XXX: IMessage
    #   (= from twisted.mail.interfaces import IMessageSMTP)
    # Message interface 정의
    implements(smtp.IMessage)

    def __init__(self):
        self.lines = []

    def lineReceived(self, line):
        self.lines.append(line)

    def eomReceived(self):
        print 'New message received:'
        print '\n'.join(self.lines)
        self.lines = None
        return defer.succeed(None)


class StdoutSMTPFactory(smtp.SMTPFactory):
    def buildProtocol(self, addr):
        proto = smtp.ESMTP()
        proto.delivery = StdoutMessageDelivery(proto)
        return proto


def run():
    log.startLogging(sys.stdout)
    reactor.listenTCP(2500, StdoutSMTPFactory())
    reactor.run()


if __name__ == '__main__':
    print 'ex.13-2 smtp_server.py'
    run()
