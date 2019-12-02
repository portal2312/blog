# -*- coding:utf8 -*-
from zope.interface import implements, Interface


from twisted.cred import checkers, credentials, portal
from twisted.internet import protocol, reactor  # protocol.Factory
from twisted.protocols import basic
# XXX: custom
from twisted.python import log

# XXX: checkers.FilePasswordDB(hash=setHash)
import hashlib

# XXX: ex.9-2
from twisted.enterprise import adbapi
from db_checker import DBCredentialsChecker


# XXX: checkers.FilePasswordDB(hash=setHash)
def setHash(username='', passwd='', passwdHash=''):
    # XXX: DB passwd 인증시 먼저 passwd 가 hash로 등록되어 있어야 함.
    return hashlib.md5(passwd).hexdigest()


class IProtocolAvatar(Interface):
    def logout():
        '''
        Avatar 에 할당 된 사용자당 자원들을 정리함.
        '''


class EchoAvatar(object):
    implements(IProtocolAvatar)

    def logout(self):
        log.msg('EchoAvatar - logout')
        pass


class Echo(basic.LineReceiver):
    portal = None
    avatar = None
    logout = None
    username = ''  # XXX: custom

    def connectionLost(self, reason):
        if self.logout:
            self.logout()
            self.avatar = None
            self.logout = None

    def lineReceived(self, line):
        if not self.avatar:
            username, passwd = line.strip().split(' ')
            log.msg('tyrLogin(%s, %s)' % (username, passwd))  # XXX: custom
            self.tryLogin(username, passwd)
        else:
            log.msg('[%s] %r' % (self.username, line))  # XXX: custom
            self.sendLine(line)

    def tryLogin(self, username, passwd):
        print username, passwd
        self.username = username  # XXX: custom
        self.portal.login(
            # XXX: ex.9-1
            credentials.UsernamePassword(
                username,
                passwd
            ),
            # XXX: ex.9-2
            # credentials.UsernameHashedPassword(
            #     username,
            #     setHash(username, passwd)
            # ),
            None,
            IProtocolAvatar
        ).addCallbacks(
            self._cbLogin,
            self._ebLogin
        )

    def _cbLogin(self, (interface, avatar, logout)):
        self.avatar = avatar
        self.logout = logout

        # XXX: custom
        log.err('(%r) Login successful, please proceed.' % self.username)
        self.sendLine(
            '(%r) Login successful, please proceed.' % self.username
        )

    def _ebLogin(self, failure):
        # XXX: custom
        log.err('(%s) Login denied, goodbye. %s' % (self.username, failure))
        self.sendLine(
            '(%s) Login denied, goodbye. %s' % (self.username, failure)
        )

        self.transport.loseConnection()


class EchoFactory(protocol.Factory):
    def __init__(self, portal):
        self.portal = portal

    def buildProtocol(self, addr):
        proto = Echo()
        proto.portal = self.portal
        return proto


class Realm(object):
    # portal.IRealm 의 class interface 갖고 와서 현재 Realm class에 정의하기.
    implements(portal.IRealm)

    def requestAvatar(self, avatarId, mind, *interfaces):
        print avatarId, mind, interfaces
        log.msg('%r %r %r' % (avatarId, mind, interfaces))
        if IProtocolAvatar in interfaces:
            avatar = EchoAvatar()
            return IProtocolAvatar, avatar, avatar.logout
        raise NotImplementedError(
            'This realm only supports the IProtocolAvatar interface.'
        )


def run():
    realm = Realm()  # 사용자 정보 받기
    myPortal = portal.Portal(realm)  # Portal (사용자 로그인) 준비
    # checker (사용자 인증) 준비
    # XXX: ex.9-1
    checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
    # XXX: ex.9-1-2 checkers.FilePasswordDB(hash=setHash)
    # checker = checkers.FilePasswordDB(
    #     'passwd.txt',
    #     delim=b':',
    #     usernameField=0,
    #     passwordField=1,
    #     caseSensitive=True,
    #     hash=setHash,
    #     cache=False
    # )
    # XXX: ex.9-2
    # dbpool = adbapi.ConnectionPool(
    #     'sqlite3',
    #     '/home/mkkim/twisted_example/chapter_8/users.db',
    #     check_same_thread=False  # XXX: need
    # )

    # checker 할 임의 데이터 추가
    # XXX: ex.9-1
    # checker.addUser('user', 'passwd')
    # checker.addUser('mkkim', 'passwd')  # XXX: custom
    # XXX: ex.9-2
    # checker = DBCredentialsChecker(
    #     dbpool.runQuery,
    #     'select userid, passwd from users where userid = ?'  # sqlite query
    # )

    # XXX: Protal 시도시 checker 등록
    myPortal.registerChecker(checker)

    # XXX: custom
    log.startLogging(file=open('echo_cred.log', 'w'))

    reactor.listenTCP(8000, EchoFactory(myPortal))
    reactor.run()


if __name__ == '__main__':
    print 'ex.9-1 echo_cred.py'
    run()


# EOF
