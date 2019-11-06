# -*- coding:utf8 -*-
'''
TODO
====
python irc_echo_bot.py ip port nickname channel


ISS
====
1. bot 실행시 아무런 반응이 없음
  - self.lineReceived 를 이용해 인증(username, passwd) 문제인 것을 확인함

  - Solve
    - set instance variable:
      @ivar username
      @ivar password

'''
import sys

from twisted.internet import reactor, protocol
from twisted.words.protocols import irc
from twisted.python import log


class EchoBot(irc.IRCClient):
    # nickname = 'echobot'  # XXX: ex.12-2

    def __init__(self, nickname):
        # XXX: ex.12-2
        self.username = nickname
        self.password = nickname
        self.nickname = nickname

    # def lineReceived(self, line):  # XXX: ex.12-2
    #     print line

    def signedOn(self):
        log.msg('signedOn: ', self.factory.channel)
        # bot IRCserver connected call
        self.join(self.factory.channel)

    def privmsg(self, user, channel, msg):
        # XXX: 확인하기
        log.msg(str({
            'user': user,
            'channel': channel,
            'msg': msg,
            'nickname': self.nickname
        }))

        # bot 어떤 msg 를 받더라도 호출함.
        # msg = private OR channel 로 부터 옴.
        user = user.split('!', 1)[0]

        if msg.startswith(self.nickname + ':'):
            log.msg('--1')
            # this msg 는 bot 의 nickname 으로 시작됨.
            # 이는 bot 에게 말한 msg 이르모 그대로 재전송.
            self.msg(channel, user + ':' + msg[len(self.nickname + ':'):])
        elif channel[1:] == self.nickname:
            log.msg('--2')
            # channel == '#echobot'
            # this msg 는 bot 에게 온 private msg 이므로 그대로 전달.
            self.msg(user, msg)
        else:
            log.msg('--3')
            pass

    def action(self, user, channel, data):
        # channel의 user가 어떤 행동을 취할 때 호출(e.g., '/me dances')
        # 이 때 bot 은 흉내를 냄.
        # TODO: error 남
        log.msg(channel)
        log.msg(data)
        self.decribe(channel, data)


class EchoBotFactory(protocol.ClientFactory):
    def __init__(self, channel):  # XXX: ex.12-2
        self.channel = channel

    def buildProtocol(self, addr):
        proto = EchoBot(self.channel)  # XXX: ex.12-2
        proto.factory = self
        return proto

    def clientConnectionLost(self, connecter, reason):
        # 접속이 끊긴 경우 재연결을 시도.
        connecter.connect()

    def clientConnectionFailed(self, connecter, reason):
        print 'connection failed:', reason
        reactor.stop()


def run():
    log.startLogging(sys.stdout)

    ip = sys.argv[1]
    port = int(sys.argv[2])
    channel = sys.argv[3]

    # XXX: channel = 'mkkim'
    f = EchoBotFactory(channel)
    reactor.connectTCP(ip, port, f)
    reactor.run()


if __name__ == '__main__':
    print 'ex.12-1 irc_echo_bot.py'
    run()
