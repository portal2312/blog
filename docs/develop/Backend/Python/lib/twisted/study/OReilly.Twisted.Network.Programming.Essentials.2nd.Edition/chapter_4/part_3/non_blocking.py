#!/usr/bin/env python
# -*- coding:utf8 -*-
'''
twisted.internet.task.deferLater

- Backend 는 각각 개별 5초 간격으로 됨. (정상)

- Web Browser 는 Browser 특성으로 인해 순서대로 5초 간격으로 됨. (비정상)

'''
from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.internet.task import deferLater
from twisted.web.server import Site, NOT_DONE_YET

import time


class BusyPage(Resource):
    isLeaf = True

    def _delayedRender(self, request):
        request.write('Finally done, at %s' % time.asctime())
        request.finish()

    def render_GET(self, request):
        d = deferLater(reactor, 5, lambda: request)
        d.addCallback(self._delayedRender)
        return NOT_DONE_YET


def run():
    factory = Site(BusyPage())
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == '__main__':
    print 'ex.4-9 blocking.py'
    run()
