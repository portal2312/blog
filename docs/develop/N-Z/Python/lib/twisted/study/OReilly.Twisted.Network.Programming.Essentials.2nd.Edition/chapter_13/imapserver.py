# -*- coding:utf8 -*-
'''
              [Protocol]
                - IMAPFactory
                  - IMAPServerProtocol
              -->
              <--
[Checker] --> [Portal] --> [Realm]
          <--          <--   - MailUserRealm
                               - IMAPUserAccount
                                 - IMAPMailbox
                                   - ExtendMaildir
                                   - MaildirMessage



'''
import email
import os
import random
from StringIO import StringIO
import sys
from zope.interface import implements
from twisted.cred import checkers, portal
from twisted.internet import reactor, protocol
from twisted.mail import imap4, maildir
from twisted.python import log


class IMAPUserAccount(object):
    '''
    - Avatar 역할
      - 사용자가 할 수 있는 행동과 데이터를 말함
    '''
    implements(imap4.IAccount)

    def __init__(self, userDir):
        self.dir = userDir

    def _getMailbox(self, path):
        '''
        mailbox 를 가져온다
        '''
        fullPath = os.path.join(self.dir, path)
        if not os.path.exists(fullPath):
            raise KeyError('No such mailbox')
        return IMAPMailbox(fullPath)

    def listMailboxes(self, ref, wildcard):
        """
        mailboxes의 모든 목록을 가져온다, 어떤 조건이 일치할 시
        """
        for box in os.listdir(self.dir):
            # XXX: box == 'Inbox'
            # XXX: yield:
            # 반복문 종료시키지 않고 결과를 반환
            # memory 큰 작업에 유용
            yield box, self._getMailbox(box)

    def select(self, path, rw=False):
        # [IMAPServerProtocol,0,127.0.0.1] [IMAPMailbox] 기본설정
        # [-] Server:  * LIST () "." "Inbox"             Protocol
        # [-] Server:  0003 OK LIST completed            Protocol
        # [-] Client:  0004 SELECT Inbox                 Protocol
        # [IMAPServerProtocol,0,127.0.0.1] [IMAPUserAccount.select]
        log.msg('[IMAPUserAccount.select]')

        return self._getMailbox(path)


class ExtendMaildir(maildir.MaildirMailbox):
    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]


class IMAPMailbox(object):
    implements(imap4.IMailbox)  # twisted.mail.interfaces.IMailboxIMAP

    def __init__(self, path):
        # XXX: path를 확장
        self.maildir = ExtendMaildir(path)
        log.msg('[IMAPMailbox] %r, %r' % (path, self.maildir))
        # listener = twisted.mail.interfaces.IMailboxIMAPListener
        #            이벤트 모음
        self.listeners = []
        self.uniqueValidityIdentifier = random.randint(1000000, 9999999)

    def getHierarchicalDelimiter(self):
        '''
        구분 기호 dot '.'
        '''
        return '.'

    def getFlags(self):
        return []

    def getMessageCount(self):
        return len(self.maildir)

    def getRecentCount(self):
        return 0

    def isWriteable(self):
        return False

    def getUIDValidity(self):
        return self.uniqueValidityIdentifier

    def _seqMessageSetToSeqDict(self, messageSet):
        '''
        maildir 안의 파일 목록을 dict로 반환

        @params return {seq: filename}
                       {0: file1, 1: file2, ..}
        '''
        if not messageSet.last:
            messageSet.last = self.getMessageCount()

        seqMap = {}
        for messageNum in messageSet:
            if messageNum >= 0 and messageNum <= self.getMessageCount():
                seqMap[messageNum] = self.maildir[messageNum-1]
        return seqMap

    def fetch(self, messages, uid):
        # 제일 마지막에 호출 됨
        log.msg('[IMAPMailbox.fetch] %r\n%r' % (messages, uid))
        '''
        하나 또는 여러 메시지를 다시 찾는다.
        '''
        if uid:
            # XXX: NotImplementedError
            # 꼭 작성해야 하는 부분이 구현되지 않았을 경우 일부러 오류내기 위해
            raise NotImplementedError(
                'This server only supports lookup by sequence number'
            )

        messagesToFetch = self._seqMessageSetToSeqDict(messages)
        for seq, filename in messagesToFetch.items():
            # XXX: seq 문서 수 0,1,2 ..
            yield seq, MaildirMessage(file(filename).read())

    def addListener(self, listener):
        self.listeners.append(listener)

    def removeListener(self, listener):
        self.listeners.remove(listener)


class MaildirMessage(object):
    implements(imap4.IMessage)  # twisted.mail.interfaces.IMessageIMAPPart

    def __init__(self, messageData):
        # file.read() 의 내용을 email.message 형태로 변경
        self.message = email.message_from_string(messageData)

    def getHeaders(self, negate, *names):
        '''
        tiwsted.mail.interfaces.IMessageIMAPPart
        '''
        if not names:
            names = self.message.keys()
        headers = {}
        if negate:
            for header in self.message.keys():
                if header.upper() not in names:
                    headers[header.lower()] = self.message.get(header, '')
        else:
            for name in names:
                headers[name.lower()] = self.message.get(name, '')
        return headers

    def getBodyFile(self):
        # message 내용의 body 내용을 가져온다
        return StringIO(self.message.get_payload())

    def isMultipart(self):
        # 가르키다, 해당 메시지 하위가 있는지 아닌지
        return self.message.is_multipart()


class MailUserRealm(object):
    implements(portal.IRealm)

    def __init__(self, baseDir):
        self.baseDir = baseDir

    def requestAvatar(self, avatarId, mind, *interfaces):
        log.msg('[requestAvatar] interfaces.keys() = %r' % (interfaces, ))
        if imap4.IAccount not in interfaces:
            raise NotImplementedError(
                'This realm only supports the imap4.IAccount interface.'
            )

        userDir = os.path.join(self.baseDir, avatarId)
        log.msg('[requestAvatar] userDir: %r' % userDir)
        avatar = IMAPUserAccount(userDir)
        # @returns: interface, avatarAspect, logout(= objects)
        return imap4.IAccount, avatar, lambda: None


class IMAPServerProtocol(imap4.IMAP4Server):
    def lineReceived(self, line):
        # def 명을 override 하고 원래 기능도 사용
        # 즉, 함수 원래 기능에 소스를 추가한 결과를 얻음
        print 'Client: ', line
        imap4.IMAP4Server.lineReceived(self, line)

    def sendLine(self, line):
        imap4.IMAP4Server.sendLine(self, line)
        print 'Server: ', line


class IMAPFactory(protocol.Factory):
    def __init__(self, portal):
        self.portal = portal

    def buildProtocol(self, addr):
        proto = IMAPServerProtocol()
        proto.portal = self.portal
        return proto


def run():
    log.startLogging(sys.stdout)

    dataDir = sys.argv[1]
    if dataDir:
        dataDir = os.path.abspath(dataDir)

    realm = MailUserRealm(dataDir)
    portal_ = portal.Portal(realm)
    checker = checkers.FilePasswordDB(
        os.path.join(dataDir, 'passwds.txt')
    )
    portal_.registerChecker(checker)

    reactor.listenTCP(8000, IMAPFactory(portal_))
    reactor.run()


if __name__ == '__main__':
    print 'ex.13-4 imapserver.py'
    run()


# EOF
