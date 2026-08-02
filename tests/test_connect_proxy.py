import unittest

from scripts.user_connect_proxy import parse_connect_target, select_ipv4


class ConnectProxyTests(unittest.TestCase):
    def test_select_ipv4_skips_cname_and_ipv6(self) -> None:
        self.assertEqual(
            select_ipv4(["example.edge.invalid.", "2001:db8::1", "203.0.113.7"]),
            "203.0.113.7",
        )

    def test_select_ipv4_fails_without_address(self) -> None:
        with self.assertRaisesRegex(ValueError, "no IPv4"):
            select_ipv4(["example.edge.invalid."])

    def test_connect_target_allows_https_only(self) -> None:
        self.assertEqual(
            parse_connect_target(b"github.com:443", frozenset({443})),
            ("github.com", 443),
        )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            parse_connect_target(b"github.com:22", frozenset({443}))

    def test_connect_target_rejects_malformed_authority(self) -> None:
        for target in (b"github.com", b":443", b"github .com:443"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    parse_connect_target(target, frozenset({443}))


if __name__ == "__main__":
    unittest.main()
