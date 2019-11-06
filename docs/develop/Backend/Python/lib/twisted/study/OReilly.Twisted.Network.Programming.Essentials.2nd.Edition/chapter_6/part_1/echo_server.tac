# -*- coding:utf8 -*-
from twisted.application import internet, service
from echo import EchoFactory
# from twisted_example.chapter_6.part_1.echo import EchoFactory


print 'ex.6-3 echo_sever.tac'
application = service.Application('echo')
echoService = internet.TCPServer(8000, EchoFactory())
echoService.setServiceParent(application)
