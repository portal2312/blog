# -*- coding:utf8 -*-
import sys

from email.mime.text import MIMEText

from twisted.internet import reactor
from twisted.mail.smtp import sendmail
from twisted.python import log


def run():
    log.startLogging(sys.stdout)

    host = 'localhost'  # 'aspmx.l.google.com'
    sender = 'sender@gmail.com'
    recipients = ['recipient@localhost']  # ['mkkim100227@gmail.com']

    msg = MIMEText('''
    3333
    Twisted is helping
    Forge e-mails to you!
    ''')
    msg['Subject'] = 'Subject3'
    msg['From'] = '"Sender <%s>"' % (sender,)
    msg['To'] = ', '.join(recipients)
    port = 8001
    deferred = sendmail(host, sender, recipients, msg.as_string(), port=port)
    deferred.addBoth(lambda result: reactor.stop())
    reactor.run()


if __name__ == '__main__':
    print 'ex.13-1 sendmail.py'
    run()
