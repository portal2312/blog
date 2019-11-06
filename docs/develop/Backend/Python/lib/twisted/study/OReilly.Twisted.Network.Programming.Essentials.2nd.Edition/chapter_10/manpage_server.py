# -*- coding:utf8 -*-
import sys


from twisted.internet import protocol, utils, reactor
from twisted.protocols.basic import LineReceiver
from twisted.python import log


class RunCommand(LineReceiver):
    def writeSuccessResponse(self, result):
        '''
        custom def
        '''
        self.transport.write(result)
        self.transport.loseConnection()

    def lineReceived(self, line):
        log.msg('Man pages requested for: %s' % (line,))
        commands = line.strip().split(' ')
        print commands
        output = utils.getProcessOutput('man', commands, errortoo=True)
        print output
        output.addCallback(self.writeSuccessResponse)


class RunCommandFactory(protocol.Factory):
    def buildProtocol(self, addr):
        return RunCommand()


def run():
    log.startLogging(sys.stdout)
    reactor.listenTCP(8000, RunCommandFactory())
    reactor.run()


if __name__ == '__main__':
    print 'ex.10-3 manpage_server.py'
    run()
