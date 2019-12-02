#!/usr/bin/env python
# -*- coding:utf8 -*-
from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.web.server import Site

import time


class BusyPage(Resource):
    isLeaf = True

    def render_GET(self, request):
        time.sleep(5)
        return 'Finally done, at %s' % (time.asctime(), )


def run():
    factory = Site(BusyPage())
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == '__main__':
    print 'ex.4-9 blocking.py'
    run()
