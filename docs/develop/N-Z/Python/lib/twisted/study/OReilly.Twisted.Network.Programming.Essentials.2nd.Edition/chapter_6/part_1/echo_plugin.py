# -*- coding:utf8 -*-
from zope.interface import implements


from twisted.application.service import IServiceMaker
from twisted.application import internet
from twisted.plugin import IPlugin
from twisted.python import usage


from twisted_example.chapter_6.part_1.echo import EchoFactory


class Options(usage.Options):
    optParameters = [
        ['port', 'p', 8000, 'The port number to listen on.'],
    ]


class EchoServiceMaker(object):
    implements(IServiceMaker, IPlugin)
    tapname = 'echo'
    description = 'A TCP-based echo server.'
    options = Options

    def makeService(self, options):
        '''
        Construct a TCPServer from a factory defined in echo.py.
        '''
        return internet.TCPServer(int(options['port']), EchoFactory())


serviceMaker = EchoServiceMaker()

print 'ex.6-4 echo_plugin.py'
