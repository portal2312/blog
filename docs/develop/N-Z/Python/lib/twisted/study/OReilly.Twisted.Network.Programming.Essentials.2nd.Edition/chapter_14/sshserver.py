# -*- coding:utf8 -*-
from twisted.conch import avatar, recvline
from twisted.conch.interfaces import IConchUser, ISession
from twisted.conch.ssh import factory, keys, session
from twisted.conch.insults import insults
from twisted.cred import portal, checkers
from twisted.internet import reactor
from zope.interface import implements

# XXX: logging
import sys
from twisted.python import log


class SSHDemoProtocol(recvline.HistoricRecvLine):
    def __init__(self, user):
        log.msg('[SSHDemoProtocol] user: %s' % user)
        self.user = user

    def connectionMade(self):
        recvline.HistoricRecvLine.connectionMade(self)
        # XXX:
        # implementer(ITerminalProtocol)
        # - class insults.TerminalProtocol
        #     --> def makeConnection(self, terminal)
        #   - class recvline.RecvLine
        #     - class recvline.HistoricRecvLine
        #
        # terminal == transport
        self.terminal.write('Welcome to my test SSH server.')
        self.terminal.nextLine()
        self.do_help()
        self.showPrompt()

    def showPrompt(self):
        self.terminal.write('$ ')

    def getCommandFunc(self, cmd):
        return getattr(self, 'do_' + cmd, None)

    def lineReceived(self, line):
        line = line.strip()
        if line:
            cmdAndArgs = line.split()
            log.msg('[SSHDemoProtocol] %s' % cmdAndArgs)
            cmd = cmdAndArgs[0]
            args = cmdAndArgs[1:]
            func = self.getCommandFunc(cmd)
            if func:
                try:
                    func(*args)
                except Exception as e:
                    self.terminal.write('Error: %s' % e)
                    self.terminal.nextLine()
            else:
                self.terminal.write('No such command.')
                self.terminal.nextLine()

        self.showPrompt()

    def do_help(self):
        pubMethods = filter(lambda n: n.startswith('do_'), dir(self))
        commands = map(lambda n: n.replace('do_', '', 1), pubMethods)
        self.terminal.write('[Help]\nCommands: ' + ' '.join(commands))
        self.terminal.nextLine()

    def do_echo(self, *args):
        self.terminal.write(' '.join(args))
        self.terminal.nextLine()

    def do_whoami(self):
        log.msg('[do_whoami]')
        self.terminal.write(self.user.username)
        self.terminal.nextLine()

    def do_quit(self):
        log.msg('[do_quit]')
        self.terminal.write('Thanks for playing.')
        self.terminal.nextLine()
        self.terminal.lostConnection()  # XXX: 안됨

    def do_clear(self):
        self.terminal.reset()


class SSHDemoAvatar(avatar.ConchUser):
    implements(ISession)

    def __init__(self, username):
        log.msg('[SSHDemoAvatar] username: %s' % username)
        avatar.ConchUser.__init__(self)
        self.username = username
        # XXX:
        # self = ConchUser
        # self.channelLookup = {}
        # channelLookup은 channel 유형을 저장
        self.channelLookup.update({'session': session.SSHSession})

    def openShell(self, protocol):
        """
        Open a shell and connect it to proto.

        @param proto: a L{ProcessProtocol} instance.
        """
        # XXX: ISession.openShell
        #      shell 열고 porotocol 연결함

        # XXX:
        # SSHDemoProtocol <- self 로 인자 넣음
        serverProtocol = insults.ServerProtocol(SSHDemoProtocol, self)

        # XXX:
        #   - tiwsted.internet.protocol.BaseProtocol
        #     --> def makeConnection(self, transport)
        #
        #     - tiwsted.internet.protocol.Protocol
        #       - twisted.conch.insults.insults.ServerProtocol
        serverProtocol.makeConnection(protocol)

        # XXX:
        # session.wrapProtocol == class _DummyTransport()
        protocol.makeConnection(session.wrapProtocol(serverProtocol))

    def getPty(self, terminal, windowSize, attrs):
        return None

    def execCommand(self, protocol, cmd):
        raise NotImplementedError()

    def closed(self):
        pass


class SSHDemoRealm(object):
    implements(portal.IRealm)

    def requestAvatar(self, avatarId, mind, *interfaces):
        if IConchUser in interfaces:
            log.msg('[SSHDemoRealm][requestAvatar]')
            log.msg(interfaces[0])
            return interfaces[0], SSHDemoAvatar(avatarId), lambda: None
        else:
            raise NotImplementedError('No supported interfaces found.')


def getRSAKeys():
    with open('/home/mkkim/.ssh/id_rsa') as privateBlobFile:
        privBlob = privateBlobFile.read()
        privKey = keys.Key.fromString(privBlob)

    with open('/home/mkkim/.ssh/id_rsa.pub') as publicBlobFile:
        pubBlob = publicBlobFile.read()
        pubKey = keys.Key.fromString(pubBlob)

    return privKey, pubKey


def run():
    log.startLogging(sys.stdout)

    sshFactory = factory.SSHFactory()
    sshFactory.portal = portal.Portal(SSHDemoRealm())

    users = {'admin': 'passwd', 'guest': 'guest'}
    sshFactory.portal.registerChecker(
        checkers.InMemoryUsernamePasswordDatabaseDontUse(**users)
    )
    privKey, pubKey = getRSAKeys()
    sshFactory.privateKeys = {'ssh-rsa': privKey}
    sshFactory.publicKeys = {'ssh-rsa': pubKey}
    reactor.listenTCP(8000, sshFactory)
    reactor.run()


if __name__ == '__main__':
    print 'ex.14-1. sshserver.py'
    run()
