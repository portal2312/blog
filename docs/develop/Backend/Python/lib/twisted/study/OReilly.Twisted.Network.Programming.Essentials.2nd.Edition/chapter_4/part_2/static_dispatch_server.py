#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet import reactor
from twisted.web.server import Site
from twisted.web.static import File


def run():
    root = File('/opt/hts')
    root.putChild('netcruz', File('/opt/hts/netcruz'))

    # 빈 /opt/hts/tmp 파일을 만들어
    # /tmp 를 추가하면
    # 나중에 결과로 나오는 tmp 선택시 /tmp 로 이동하게 됨.
    # 마치 simbolic link 와 비슷.
    root.putChild('tmp', File('/tmp'))
    factory = Site(root)
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == '__main__':
    print 'Example 4-4 static_dispatch_server.py'
    run()


# EOF
