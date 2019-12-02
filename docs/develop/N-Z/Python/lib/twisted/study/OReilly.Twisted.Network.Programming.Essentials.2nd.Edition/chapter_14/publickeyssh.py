# -*- coding:utf8 -*-
from sshserver import SSHDemoRealm, getRSAKeys
from twisted.conch import error
from twisted.conch.ssh import keys, factory
from twisted.cred import checkers, credentials, portal
from twisted.internet import reactor
from twisted.python import failure
from zope.interface import implements
import base64


class PublicKeyCredentialsChecker(object):
    implements(checkers.ICredentialsChecker)
    credentialInterfaces = (credentials.ISSHPrivateKey, )

    def __init__(self, authorizedKeys):
        self.authorizedKeys = authorizedKeys

    def requestAvatarId(self, credentials):
        print 'credentials:', credentials.__dict__
        userKeyString = self.authorizedKeys.get(credentials.username)

        if not userKeyString:
            return failure.Failure(error.ConchError('No such user'))

        # decoding 전에 userKeyString 앞에 붙는 'ssh-rsa' 문자열을 제거한다.
        if credentials.blob != base64.decodestring(
                userKeyString.split(' ')[1]):
            raise failure.Failure(
                error.ConchError('I do not recognize that key')
            )

        if not credentials.signature:
            raise failure.Failure(error.ValidPublicKey())

        userKey = keys.Key.fromString(data=userKeyString)

        if userKey.verify(credentials.signature, credentials.sigData):
            return credentials.username
        else:
            print 'signature check failed'
            return failure.Failure(
                error.ConchError('Incorrect signature')
            )


def run():
    sshFactory = factory.SSHFactory()
    sshFactory.portal = portal.Portal(SSHDemoRealm())

    # Server
    privKey, pubKey = getRSAKeys()
    sshFactory.privateKeys = {'ssh-rsa': privKey}
    sshFactory.publicKeys = {'ssh-rsa': pubKey}

    # Client - try connect ssh keys
    authorizedKeys = {
        'admin': 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDAuHXSNRydis7q+BZrHAXStGDotfT+kzARk3zZM3ytuprlMB9hZsfE7i4ee2U3GczstgImxK0npJTAamOdW1UZGiePPEvMp5lB6J5HDk+qZ1BWaJdCN3XWW5LUfAoCmR6INEA2Fzeh1gowSR1p0ZIqUymIFdprBx7hk7piQVkvWoUvQyf12Eg0LpxHorktHHZg0LJyS55unMRTR/bHgRe0SYygzLjTyyn1/l0RD1bfKwletqG/uVCxXTafp6VSnQJ1ajgQGw3XGRj3xZpbs/+KjQjddxWLAXXK1IGCFIY6r4PS93BBCzdTGWxHBWTQAJsPDOsFwFbwsQ+PFJaet6N9 mkkim@localhost.localdomain'
    }

    sshFactory.portal.registerChecker(
        PublicKeyCredentialsChecker(authorizedKeys)
    )

    reactor.listenTCP(8000, sshFactory)
    reactor.run()


if __name__ == '__main__':
    print 'ex.14-2. publickeyssh.py'
    run()
