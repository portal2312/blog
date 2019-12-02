# -*- coding:utf8 -*-
from twisted.internet import reactor
from twisted.web import server, resource
from twisted.cred import portal, checkers
from twisted.conch import manhole, manhole_ssh
from twisted.python import log


class LinksPage(resource.Resource):
    # XXX: isLeaf
    # 마지막 노드인지 여부를 결정
    # 1 == True
    # getChildWithDefault 를 호출하지 않음
    isLeaf = 1

    def __init__(self, links):
        log.msg('links: %s' % links)

        resource.Resource.__init__(self)
        self.links = links

    def render(self, request):
        log.msg('[render] request: %s' % (request.__dict__))

        return '<ul>%s</ul>' % (
            ''.join([
                '<li><a href=\'%s\'>%s</a></li>' % (link, title)
                for title, link in self.links.items()
            ]),
        )


def getManholeFactory(namespace, **users):
    '''
    Traceback (most recent call last):
      File "manholeserver.py", line 81, in <module>
        run()
      File "manholeserver.py", line 75, in run
        reactor.listenTCP(8001, getManholeFactory(globals(), mkkim='passwd'))
      File "/usr/local/lib/pyenv/versions/mkkim-virenv-2.7.13/lib/python2.7/site-packages/twisted/internet/posixbase.py", line 478, in listenTCP
        p.startListening()
      File "/usr/local/lib/pyenv/versions/mkkim-virenv-2.7.13/lib/python2.7/site-packages/twisted/internet/tcp.py", line 1001, in startListening
        self.factory.doStart()
      File "/usr/local/lib/pyenv/versions/mkkim-virenv-2.7.13/lib/python2.7/site-packages/twisted/internet/protocol.py", line 76, in doStart
        self.startFactory()
      File "/usr/local/lib/pyenv/versions/mkkim-virenv-2.7.13/lib/python2.7/site-packages/twisted/conch/ssh/factory.py", line 41, in startFactory
        raise error.ConchError('no host keys, failing')
    twisted.conch.error.ConchError: ('no host keys, failing', None)
    '''
    log.msg('[getManholeFactory]')
    # XXX: namespace = globals(), users = {'admin': 'aaa'}
    log.msg(namespace)
    log.msg(users)

    # XXX: Realm
    realm = manhole_ssh.TerminalRealm()

    # XXX:
    def getManhole(_): return manhole.Manhole(namespace)

    # XXX:
    # - twisted.internet.protocol.BaseProtocol
    #   - twisted.internet.protocol.Protocol
    #     - twisted.conch.insults.insults.ServerProtocol
    # >>>   - @params protocolFactory
    # --------------------------------
    # - twisted.conch.manhole_ssh.TerminalRealm
    #   - @ivar chainedProtocolFactory
    #     - protocolFactory
    realm.chainedProtocolFactory.protocolFactory = getManhole

    # XXX: Checker
    checker_ = checkers.InMemoryUsernamePasswordDatabaseDontUse(**users)

    # XXX: Portal
    p = portal.Portal(realm)
    p.registerChecker(checker_)

    # XXX: Factory
    f = manhole_ssh.ConchFactory(p)
    return f


def run():
    links = {
        'Twisted': 'http://twistedmatrix.com/',
        'Python': 'http://python.org'
    }
    site = server.Site(LinksPage(links))  # Factory 역할
    reactor.listenTCP(8000, site)
    reactor.listenTCP(8001, getManholeFactory(globals(), mkkim='passwd'))
    reactor.run()


if __name__ == '__main__':
    print 'ex.14-3. manholeserver.py'
    run()
