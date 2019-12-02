#!/usr/bin/env python
# -*- coding:utf-8 -*-
import urllib2


if __name__ == '__main__':
    print 'Example 4-2 requesthandlerclient.py'

    print urllib2.urlopen('http://localhost:8000').read()
    print urllib2.urlopen('http://localhost:8000/about').read()
    print urllib2.urlopen('http://localhost:8000/warnning').read()


# EOF
