# -*- coding:utf8 -*-
'''
ClientCommandFactory(protocol.ClientFactory)  # [Factory]
  def buildProtocol():

    - ClientCommandTransport(tarnsport.SSHClientTransport)  # [Protocol]
        def connectionSecure():
          - self.requestService()  # request

            - PasswordAuth(userauth.SSHUserAuthClient)        # [Credentials]

              - ClientConnection(connection.SSHConnection)    # action
                  - def serviceStarted():
                      self.openChannel()

                        - CommandChannel(channel.SSHChannel)  # action
'''
import sys
import getpass
from twisted.conch.ssh import (
    connection,
    channel,
    common,
    transport,
    userauth,
)
from twisted.internet import defer, protocol, reactor
from twisted.python import log


class ClientCommandTransport(transport.SSHClientTransport):
    def __init__(self, username, password, command):
        self.username = username
        self.password = password
        self.command = command

    def verifyHostKey(self, hostKey, fingerprint):
        # 실제 application 에서, 이 서버에 보낸 핑거프린트가 기대하던 핑거프린트와 일치하는지 확인.
        log.msg('''[ClientCommandTransport][verifyHostKey]\n
            hostKey: %r,
            fingerprint: %r
        ''' % (hostKey, fingerprint)
        )
        return defer.succeed(True)

    def connectionSecure(self):
        # 초기 암호화된 연결이 맺어지면 호출
        log.msg('[ClientCommandTransport][connectionSecure] >>> self.requestService')

        passwordAuth = PasswordAuth(
            self.username,
            self.password,
            ClientConnection(self.command)
        )
        # XXX: Credentials
        # passwordAuth = {
        #    'instance': <__main__.ClientConnection instance at 0x7f212dfbbfc8>,
        #    'password': 'mkkim!1',
        #    'user': 'mkkim'
        # }

        # 서비스 요청
        self.requestService(passwordAuth)


class PasswordAuth(userauth.SSHUserAuthClient):
    # 인증 처리 담당
    def __init__(self, user, password, connection):
        userauth.SSHUserAuthClient.__init__(self, user, connection)
        self.password = password

    def getPassword(self, prompt=None):
        # 인증이 완료되면 로그인에 사용할 passwd를 반환
        log.msg('[PasswordAuth][getPassword] >>> self.password')
        return defer.succeed(self.password)


class ClientConnection(connection.SSHConnection):
    # 인증이 성공한 이후 연결을 관리
    def __init__(self, command, *args, **kwargs):
        connection.SSHConnection.__init__(self)
        self.command = command

    def serviceStarted(self):
        # Client 가 성공적 로그인하자마자 호출
        log.msg('[ClientConnection][serviceStarted] >>> self.openChannel')
        self.openChannel(  # channel이 준비되었을 때 호출
            CommandChannel(self.command, conn=self)
        )


class CommandChannel(channel.SSHChannel):
    # SSH Server 연결하고 인증된 채널과 함께 동작
    name = 'session'  # @ivar name: channel name

    def __init__(self, command, *args, **kwargs):
        channel.SSHChannel.__init__(self, *args, **kwargs)
        self.command = command

    def channelOpen(self, data):
        # XXX: self.conn.sendRequest - 요청을 보냄
        # self.conn = twisted.conch.ssh.connection.SSHConnection
        # self.conn.sendRequest(self, channel, requestType, data, wantReply=0):
        #   Send a request to a channel.
        #   @type channel:      subclass of C{SSHChannel}
        #   @type requestType:  L{bytes}
        #   @type data:         L{bytes}
        #   @type wantReply:    L{bool}
        #   @rtype              C{Deferred}/L{None}
        self.conn.sendRequest(  # 서버에 명령어 전송
            self,
            'exec',  # 실행
            common.NS(self.command),  # 서버에 명령어 전송을 위해 network 문자열로 formating
            wantReply=True
        ).addCallback(
            self._gotResponse
        )

    def _gotResponse(self, _):
        # XXX: self.conn.sendEOF
        # channel 에 문서의 끝(EOF) 라고 보내 연결을 닫음
        # self.conn = twisted.conch.ssh.connection.SSHConnection
        # self.conn.sendEOF(self, channel):
        # Send an EOF (End of File) for a channel.
        #   @type channel:  subclass of L{SSHChannel}
        self.conn.sendEOF(self)

    def dataReceived(self, data):
        # 서버의 데이터를 받음
        print data

    def closed(self):
        # 채널이 성공적으로 닫힘을 알게 끔 호출
        reactor.stop()


class ClientCommandFactory(protocol.ClientFactory):
    def __init__(self, username, password, command):
        self.username = username
        self.password = password
        self.command = command

    def buildProtocol(self, addr):
        proto = ClientCommandTransport(
            self.username, self.password, self.command
        )
        return proto


def run():
    log.startLogging(sys.stdout)

    info = sys.argv[1].split('@')
    username = info[0]
    server_ip = info[1]
    command = sys.argv[2]
    password = getpass.getpass('Password: ')

    reactor.connectTCP(
        server_ip,
        22,  # XXX: SSH Port (default=22)
        ClientCommandFactory(username, password, command)
    )
    reactor.run()


if __name__ == '__main__':
    print 'ex.14-4. sshclient.py'
    run()
