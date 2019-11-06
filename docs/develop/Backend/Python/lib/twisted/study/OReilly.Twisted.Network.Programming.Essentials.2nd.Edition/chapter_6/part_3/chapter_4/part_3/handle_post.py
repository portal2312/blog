#!/usr/bin/env python
# -*- coding:utf8 -*-
from twisted.internet import reactor
from twisted.web.resource import Resource
from twisted.web.server import Site

import cgi


class FormPage(Resource):
    isLeaf = True

    def render_GET(self, request):
        return '''
        <html>
        <body>
        <form method="POST">
        <input name="form-field" type="text" />
        <input type="submit" />
        </form>
        </body>
        </html>
        '''

    def render_POST(self, request):
        return '''
        <html>
        <body>
        you submitted: %s
        </body>
        </html>
        ''' % (cgi.escape(request.args['form-field'][0]),)


def run():
    factory = Site(FormPage())
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == '__main__':
    print 'ex.4-8 handle_post.py'
    run()
