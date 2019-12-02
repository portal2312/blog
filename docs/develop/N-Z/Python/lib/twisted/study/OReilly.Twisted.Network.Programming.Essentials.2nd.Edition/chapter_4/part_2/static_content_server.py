#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet import reactor
from twisted.web.server import Site
from twisted.web.static import File


def run():
    value = '/opt/hts/'  # 권한이 있는 자신의 folder 경로
    print value
    resource = File(value)
    factory = Site(resource)
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == '__main__':
    print 'Example 4-3 static_content_server.py'
    run()


# EOF
