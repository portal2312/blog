#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
- localhost 외 접속시 방화벽 추가후 하면 가능함.

- 아래와 같이 server 측 방화벽을 해제함.

> su -
#> vi /etc/sysconfig/iptables

...
# XXX: TEST by.mkkim firewall
-A RH-Firewall-1-INPUT -m state --state NEW -m tcp -p tcp --dport 8000 -j ACCEPT

...
# XXX: 여기 아래부터는 추가해도 허용 안됨.
-A RH-Firewall-1-INPUT -j REJECT --reject-with icmp-host-prohibited
COMMIT
:wq

#> /etc/init.d/iptables restart

'''
import urllib2


def run():
    print urllib2.urlopen(url='http://localhost:8000').read()


if __name__ == '__main__':
    print 'Example 4-3 static_content_server.py'
    run()


# EOF
