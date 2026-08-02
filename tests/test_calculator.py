import unittest

from jphrl.envs.calculator import evaluate_expression


class CalculatorTests(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        self.assertEqual(evaluate_expression("17 + 25"), "42")
        self.assertEqual(evaluate_expression("13 * 7"), "91")
        self.assertEqual(evaluate_expression("(8 + 4) / 3"), "4")
        self.assertEqual(evaluate_expression("5 / 2"), "5/2")

    def test_code_execution_is_forbidden(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_expression("__import__('os').system('id')")
        with self.assertRaises(ValueError):
            evaluate_expression("open('/etc/passwd').read()")

    def test_non_contract_operators_are_forbidden(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_expression("2 ** 100")
        with self.assertRaises(ValueError):
            evaluate_expression("5 // 2")
        with self.assertRaises(ValueError):
            evaluate_expression("5 % 2")

    def test_floats_and_zero_division_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_expression("0.1 + 0.2")
        with self.assertRaises(ZeroDivisionError):
            evaluate_expression("1 / 0")


if __name__ == "__main__":
    unittest.main()
