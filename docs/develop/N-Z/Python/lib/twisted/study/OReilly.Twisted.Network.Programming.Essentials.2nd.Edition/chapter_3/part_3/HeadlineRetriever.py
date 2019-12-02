#!/usr/bin/env python
# -*- coding:utf-8 -*-
from twisted.internet import defer, reactor


def printData(result):
    print 'printData: %s' % result
    reactor.stop()


def printError(failure):
    print 'printError: %s' % failure
    reactor.stop()


class HeadlineRetriever(object):
    def processHeadline(self, headline):
        if len(headline) > 50:
            self.d.errback(
                '<error> too long. (%s)' % headline
            )
        else:
            self.d.callback(headline)

    def _toHTML(self, result):
        return 'toHTML: <h>%s</h>' % result

    def getHeadline(self, input):
        print '>>> getHeadline'
        self.d = defer.Deferred()
        reactor.callLater(1, self.processHeadline, input)
        self.d.addCallback(self._toHTML)
        return self.d


def run():
    h = HeadlineRetriever()
    d = h.getHeadline('mkkim')
    d.addCallbacks(printData, errback=printError)
    reactor.run()


if __name__ == '__main__':
    print 'Example 3-4 HeadlineRetriever.py'
    run()


# EOF
