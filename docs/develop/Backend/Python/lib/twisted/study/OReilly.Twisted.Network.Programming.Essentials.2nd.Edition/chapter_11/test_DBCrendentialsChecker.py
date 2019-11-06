# -*- coding:utf8 -*-
from twisted.trial import unittest
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import reactor
from twisted.internet.defer import Deferred

from db_checker import DBCredentialsChecker


class DBCredentialsCheckerTestCase(unittest.TestCase):
    def test_requestAvatarIdGoodCredentials(self):
        '''
        올바른 크리덴셜로 requestAvatarId를 호출하면
        사용자 이름을 반환한다.
        '''
        def fakeRunqueryMatchingPassword(query, username):
            '''
            setData
            '''
            d = Deferred()
            reactor.callLater(0, d.callback, (('user', 'passwd'),))
            return d

        creds = credentials.UsernameHashedPassword('user', 'passwd')
        checker = DBCredentialsChecker(
            fakeRunqueryMatchingPassword,
            'fake query'
        )
        d = checker.requestAvatarId(creds)

        def checkRequestAvatarCb(result):
            self.assertEqual(result, 'user')

        d.addCallbacks(checkRequestAvatarCb)
        return d

    # ex.11-9
    def test_requestAvatarIdBadCredentials(self):
        '''
        bad credentials call requestAvatarId
        UnauthorizedLogin Error
        '''
        def fakeRunqueryBadPassword(query, username):
            d = Deferred()
            reactor.callLater(0, d.callback, (('user', 'badpasswd'),))
            return d

        creds = credentials.UsernameHashedPassword('user', 'passwd')
        checker = DBCredentialsChecker(
            fakeRunqueryBadPassword,
            'fake query'
        )
        d = checker.requestAvatarId(creds)

        def checkError(result):
            self.assertEqual(result.message, 'Passwd mismatch')

        return self.assertFailure(
            d, UnauthorizedLogin
        ).addCallback(
            checkError
        )

    # ex.11-9
    def test_requestAvatarIdNoUser(self):
        '''
        unknwon credentials call requestAvatarId
        UnauthorizedLogin Error
        '''
        def fakeRunqueryMissingUser(query, username):
            d = Deferred()
            reactor.callLater(0, d.callback, ())
            return d

        creds = credentials.UsernameHashedPassword('user', 'passwd')
        checker = DBCredentialsChecker(
            fakeRunqueryMissingUser,
            'fake query'
        )
        d = checker.requestAvatarId(creds)

        def checkError(result):
            self.assertEqual(result.message, 'User not in Database.')

        return self.assertFailure(
            d, UnauthorizedLogin
        ).addCallback(
            checkError
        )
