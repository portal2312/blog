# -*- coding:utf8 -*-
from zope.interface import implements

from twisted.application.service import IServiceMaker
from twisted.application import internet
from twisted.plugin import IPlugin
from twisted.python import usage

# XXX: vi ~/.bash_profile
# PYTHONPATH="$HOME/twisted_example/chapter_9/echo_cred_plugin:$PYTHONPATH"
from twisted_example.chapter_9.echo_cred import EchoFactory, Realm
from twisted.cred import credentials, portal, strcred


class Options(usage.Options, strcred.AuthOptionMixin):
    # credentials의 username 과 passwd 를 캡슐화
    # supportedInterfaces = tuple(), 안하면 인자 못 받음
    supportedInterfaces = (credentials.IUsernamePassword,)
    optParameters = [[
        'port', 'p', 8000, 'The port number to listen on.'
    ]]


class EchoServiceMaker(object):
    implements(IServiceMaker, IPlugin)
    tapname = 'echo9'
    description = 'A TCP-based echo server. (ex.9-3)'
    options = Options

    def makeService(self, options):
        '''
        EchoFactory 에서 TCPServer 를 create.
        '''
        # XXX: twistd echo9 시 error 발생은 정상이니 무시.
        p = portal.Portal(Realm(), options['credCheckers'])
        return internet.TCPServer(int(options['port']), EchoFactory(p))


print 'ex.9-3 echo_cred_plugin.py'
serviceMaker = EchoServiceMaker()
