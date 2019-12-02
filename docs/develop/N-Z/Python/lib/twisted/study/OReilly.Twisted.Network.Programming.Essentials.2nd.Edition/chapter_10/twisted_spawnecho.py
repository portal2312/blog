# -*- coding:utf8 -*-
from twisted_example.chapter_6.part_1.echo import EchoFactory

from twisted.internet import protocol, reactor


class EchoProcessProtocol(protocol.ProcessProtocol):
    def connectionMade(self):
        print 'connectionMade called.'
        reactor.callLater(10, self.terminateProcess)

    def terminateProcess(self):
        self.transport.signalProcess('TERM')

    def outReceived(self, data):
        print 'outReceived called with %d bytes of data:\n%s' % (
            len(data), data
        )

    def errReceived(self, data):
        print 'errReceived called with %d bytes of data:\n%s' % (
            len(data), data
        )

    def inConnectionLost(self):
        print 'inConnectionLost called, stdin closed.'

    def outConnectionLost(self):
        print 'outConnectionLost called, stdin closed.'

    def errConnectionLost(self):
        print 'errConnectionLost called, stdin closed.'

    def processExited(self, reason):
        print 'processExited called with status %d' % (
            (reason.value.exitCode,)
        )

    def processEnded(self, reason):
        print 'processEnded called with status %d' % (
            (reason.value.exitCode,)
        )
        print 'ALL FDs are now closed, and the process has been reaped.'
        reactor.stop()


def run():
    '''
    # OSError: [Errno 2] No such file or directory

    - commandAndArgs = [] 안의 값 확인하기.

    - 안의 값들이 참조 받을 수 있는 경로인지 확인하기.
    '''
    pp = EchoProcessProtocol()
    commandAndArgs = [
        # XXX: twistd - virtualenv 시 절대경로
        # 'twistd',
        '/usr/local/lib/pyenv/shims/twistd',
        # XXX: twistd [options]
        '-ny',
        # XXX: .tac = twistd 참조 가능여부(또는 PYTHONPATH 확인) , 절대경로
        # 안되면 현재 경로에 tac 과 import 파일을 두면 됨.
        'echo_server.tac'
        # '/home/mkkim/twisted_example/chapter_6/part_1/echo_server.tac'
    ]
    reactor.spawnProcess(pp, commandAndArgs[0], args=commandAndArgs)
    reactor.run()


if __name__ == '__main__':
    print 'ex.10-4 twisted_spawnecho.py'
    run()


# EOF
