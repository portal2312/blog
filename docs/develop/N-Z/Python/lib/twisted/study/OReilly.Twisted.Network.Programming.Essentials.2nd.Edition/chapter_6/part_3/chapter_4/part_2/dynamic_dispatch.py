#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet import reactor
from twisted.web.resource import Resource, NoResource
from twisted.web.server import Site

from calendar import calendar


class YearPage(Resource):
    # XXX: rendering 될 수 있으면서 getchild 자원 생성하기
    #      isLeaf = False (Default) > /2013/aa > 404
    #             = True > /2013/aa 시 /2013 페이지를 보여줌.
    # isLeaf = True

    def __init__(self, path):
        Resource.__init__(self)
        self.year = path

    def render_GET(self, request):
        return '''
        <html>
            <body>
                <pre>%s</pre>
            </body>
        </html>
        ''' % (calendar(self.year),)


class CalendarHome(Resource):
    def getChild(self, path, request):
        if path == '':
            return self

        if path.isdigit():
            return YearPage(int(path))
        else:
            return NoResource(message="Sorry. No luck finding that resource.")

    def render_GET(self, request):
        return '''
        <html>
            <body>
                welcome
            </body>
        </html>
        '''


def run():
    root = CalendarHome()
    factory = Site(root)
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == '__main__':
    print 'Example 4-6 dynamic_dispatch.py'
    run()


# EOF
