#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import urllib2


def main():
    result = urllib2.urlopen('http://localhost:8000')
    result = result.read()
    print result


def connectTelnet(url, port=8000):
    cmd = 'telnet %s %d' % (url, port)
    print cmd
    os.system(cmd)
    # input 'mkkim'
    # input ''
    # end


if __name__ == '__main__':
    print 'Example 4-1 webechoclient.py'
    main()
    # url = 'localhost'
    # port = 8000
    # connectTelnet(url, port)


# EOF
