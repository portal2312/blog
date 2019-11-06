#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.web.server import Site

import time


class ClockPage(Resource):
    isLeaf = True

    def render_GET(self, request):
        return 'The local time is %s.' % time.ctime()


def run():
    resource = ClockPage()
    factory = Site(resource)
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == '__main__':
    print 'Example 4-5 dynamic_content.py'
    run()


# EOF
