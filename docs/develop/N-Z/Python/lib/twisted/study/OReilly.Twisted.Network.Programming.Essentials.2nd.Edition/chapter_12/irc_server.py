# -*- coding:utf8 -*-
import sys

from twisted.cred import checkers, portal
from twisted.internet import reactor
from twisted.words import service
from twisted.python import log


def run():
    log.startLogging(sys.stdout)

    wordsRealm = service.InMemoryWordsRealm('irc_server')
    wordsRealm.createGroupOnRequest = True

    checker = checkers.FilePasswordDB('passwd.txt')
    portal_ = portal.Portal(wordsRealm, [checker])

    port = 6667
    if sys.argv[1]:
        port = int(sys.argv[1])

    reactor.listenTCP(port, service.IRCFactory(wordsRealm, portal_))
    reactor.run()


if __name__ == '__main__':
    print 'ex.12-2 irc_server.py'
    run()
