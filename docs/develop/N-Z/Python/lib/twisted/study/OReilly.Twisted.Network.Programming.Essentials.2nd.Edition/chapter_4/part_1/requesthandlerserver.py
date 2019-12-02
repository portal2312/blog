#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet import reactor
from twisted.web import http

http.HTTPChannel.request

class MyRequestHandler(http.Request):
    resources = {
        '/': '<h1>home</h1>Home Page',
        '/about': '<h1>About></1>All about me',
    }

    def process(self):
        self.setHeader('Content-Type', 'text/html')
        if self.resources.has_key(self.path):
            self.write(self.resources[self.path])
        else:
            self.setResponseCode(http.NOT_FOUND)
            self.write('<h1>Not Found</h1>Sorry, not such resource.')

        self.finish()


class MyHTTPProtocol(http.HTTPChannel):
    requestFactory = MyRequestHandler


class MyHTTPFactory(http.HTTPFactory):
    def buildProtocol(self, addr):
        return MyHTTPProtocol()


if __name__ == '__main__':
    print 'Example 4-2 requesthandlerserver.py'
    reactor.listenTCP(8000, MyHTTPFactory())
    reactor.run()


# EOF
