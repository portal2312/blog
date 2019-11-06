# -*- coding:utf8 -*-
from chatserver import ChatFactory

from twisted.test import proto_helpers
from twisted.trial import unittest


class ChatServerTestCase(unittest.TestCase):
    def setUp(self):
        self.factory = ChatFactory()
        # self.proto = protocol.Protocol
        # LineReceiver 는 protocol.Protocol 을 상속받아 사용되었음.
        self.proto = self.factory.buildProtocol(('localhost', 0))
        self.transport = proto_helpers.StringTransport()
        self.proto.makeConnection(self.transport)

    def test_connect(self):
        self.assertEqual(self.transport.value(), 'Login-name: \r\n')

    def test_logout(self):
        '''
        - mkkim 이 직접 추가
        - 내용:
        msg 까지 test 시
        두개의 계정을 접속까지 시켜 놓고
        한 계정을 logout 시
        다른 계정에서 보여지는 msg 를 확인 것
        '''
        name1 = 'mkkim3'
        self.proto.lineReceived(name1)
        self.transport.clear()

        name2 = 'mkkim4'
        proto2 = self.factory.buildProtocol(('localhost', 0))
        transport2 = proto_helpers.StringTransport()
        proto2.makeConnection(transport2)
        proto2.lineReceived(name2)
        transport2.clear()

        # protocol.Protocol 의 함수중 접속을 종료 및 잃어버리는 함수
        # Protocol 의 autocomplete 에서 찾음
        self.proto.connectionLost('')
        self.assertNotIn(name1, self.proto.factory.users)

        self.assertEqual(
            transport2.value(),
            '%s has left the channel. (%s)\r\n' % (
                name1,
                self.factory.users.keys()
            )
        )

    def test_register(self):
        self.assertEqual(self.proto.state, 'REGISTER')
        self.proto.lineReceived('mkkim')
        self.assertIn('mkkim', self.proto.factory.users)
        self.assertEqual(self.proto.state, 'CHAT')

    def test_chat(self):
        self.proto.lineReceived('mkkim')
        proto2 = self.factory.buildProtocol(('localhost', 0))
        transport2 = proto_helpers.StringTransport()
        proto2.makeConnection(transport2)

        self.transport.clear()
        proto2.lineReceived('mkkim2')

        self.assertEqual(
            self.transport.value(), '"mkkim2" has joined the channel.\r\n'
        )
