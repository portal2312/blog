# -*- coding:utf8 -*-
from zope.interface import implements

from twisted.application.service import IServiceMaker
from twisted.application import internet
from twisted.plugin import IPlugin
from twisted.python import usage

# XXX: vi ~/.bash_profile
# PYTHONPATH="$HOME/twisted_example/chapter_6/part_1/echoproject:$PYTHONPATH"
from twisted_example.chapter_6.part_1.echo import EchoFactory


class Options(usage.Options):
    optParameters = [
        ['port', 'p', 8000, 'The port number to listen on.'],
    ]


class EchoServiceMaker(object):
    implements(IServiceMaker, IPlugin)
    tapname = 'echo6'
    description = 'A TCP-based echo server. (ex.6-4)'
    options = Options

    def makeService(self, options):
        '''
        Construct a TCPServer from a factory defined in echo.py.
        '''
        return internet.TCPServer(int(options['port']), EchoFactory())


# XXX: disabled
# 사용시 twistd 에서 목록 없음.
# if __name__ == '__main__':
#     pass

print 'ex.6-4 echo_plugin.py'
serviceMaker = EchoServiceMaker()
