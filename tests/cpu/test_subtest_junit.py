"""Exercise actual pytest subprocesses, rather than fabricated JUnit counters."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from illm.domain.errors import IllmError
from illm.qualification import parse_junit
from illm.reference_qualification import _verify_pytest_transcript


ROOT = Path(__file__).resolve().parents[2]


class SubtestJUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = ROOT / "out" / "subtest-junit-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.directory = Path(self.temporary.name)
        shutil.copyfile(ROOT / "tests" / "conftest.py", self.directory / "conftest.py")
        self.addCleanup(self.temporary.cleanup)

    def _run(self, source: str, *, junit: bool = True, quiet: bool = True) -> subprocess.CompletedProcess[str]:
        path = self.directory / "test_vectors.py"
        path.write_text(source, encoding="utf-8")
        command = [sys.executable, "-m", "pytest"]
        if quiet:
            command.append("-q")
        command.extend(["-p", "no:cacheprovider", str(path)])
        if junit:
            command.extend(["--junitxml", str(self.directory / "junit.xml")])
        environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        return subprocess.run(command, cwd=self.directory, env=environment, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", check=False)

    def _suite(self) -> tuple[ET.Element, list[ET.Element]]:
        suite = ET.parse(self.directory / "junit.xml").getroot().find("testsuite")
        self.assertIsNotNone(suite)
        assert suite is not None
        cases = suite.findall("testcase")
        self.assertEqual(int(suite.attrib["tests"]), len(cases))
        for attribute, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            self.assertEqual(int(suite.attrib[attribute]), sum(case.find(tag) is not None for case in cases))
        return suite, cases

    def test_passing_vectors_and_parent_are_separate_cases_with_unique_contexts(self) -> None:
        process = self._run('''import unittest
class Vectors(unittest.TestCase):
    def test_vectors(self):
        for value in (1, 1, 2):
            with self.subTest("real vector", value=value):
                self.assertGreater(value, 0)
def test_ordinary():
    assert True
''')
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        suite, cases = self._suite()
        self.assertEqual(int(suite.attrib["tests"]), 5)
        names = [case.attrib["name"] for case in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("test_vectors", names)
        self.assertIn("test_ordinary", names)
        subtests = [name for name in names if "illm-subtest#" in name]
        self.assertEqual(len(subtests), 3)
        self.assertTrue(all("real vector" in name and "value" in name for name in subtests))
        self.assertEqual(parse_junit(self.directory / "junit.xml")["tests"], 5)

    def test_failed_vector_is_preserved_and_process_fails(self) -> None:
        process = self._run('''import unittest
class Vectors(unittest.TestCase):
    def test_vectors(self):
        for value in (1, -1, 2):
            with self.subTest(value=value):
                self.assertGreater(value, 0, "actual-vector-failure")
''', quiet=False)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        self.assertRegex(process.stdout, r"(?m)^=+ 1 failed, 1 passed in [0-9.]+s =+$")
        suite, cases = self._suite()
        self.assertEqual(len(cases), 4)
        self.assertGreater(int(suite.attrib["failures"]), 0)
        self.assertTrue(any("actual-vector-failure" in (case.findtext("failure") or "") for case in cases))
        with self.assertRaises(IllmError):
            parse_junit(self.directory / "junit.xml")

    def test_skipped_vector_and_ordinary_skip_are_preserved(self) -> None:
        process = self._run('''import unittest
import pytest
class Vectors(unittest.TestCase):
    def test_vectors(self):
        with self.subTest(reason="unavailable"):
            self.skipTest("actual-vector-skip")
        with self.subTest(reason="available"):
            self.assertTrue(True)
@pytest.mark.skip(reason="actual-parent-skip")
def test_ordinary_skip():
    assert False
''')
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        suite, cases = self._suite()
        self.assertEqual(len(cases), 4)
        self.assertEqual(int(suite.attrib["skipped"]), 2)
        self.assertEqual(parse_junit(self.directory / "junit.xml")["skipped"], 2)

    def test_no_junit_does_not_change_success_or_failure(self) -> None:
        source = '''import unittest
class Vectors(unittest.TestCase):
    def test_vectors(self):
        with self.subTest(value=1):
            self.assertTrue(VECTOR_VALUE)
'''
        for value, expected in (("True", 0), ("False", 1)):
            with self.subTest(value=value):
                process = self._run(source.replace("VECTOR_VALUE", value), junit=False)
                self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
                self.assertFalse((self.directory / "junit.xml").exists())

    def test_builtin_subtests_fixture_keeps_its_executed_reports(self) -> None:
        process = self._run('''def test_vectors(subtests):
    for value in ("same", "same"):
        with subtests.test("fixture vector", value=value):
            assert value == "same"
''')
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        _, cases = self._suite()
        self.assertEqual(len(cases), 3)
        self.assertEqual(parse_junit(self.directory / "junit.xml")["tests"], 3)

    def test_gate_summary_cross_checks_parent_and_subtest_counts(self) -> None:
        process = self._run('''import unittest
import pytest
class Vectors(unittest.TestCase):
    def test_vectors(self):
        for value in (1, 1, 2):
            with self.subTest("real vector", value=value):
                self.assertGreater(value, 0)
def test_ordinary():
    assert True
@pytest.mark.skip(reason="parent-only-skip")
def test_parent_skip():
    assert False
''', quiet=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        counts = _verify_pytest_transcript(process.stdout, self.directory / "junit.xml")
        self.assertEqual(counts, {"parent_tests": 3, "parent_passed": 2, "parent_skipped": 1,
                                  "subtests": 3, "subtests_passed": 3})
        with self.assertRaisesRegex(IllmError, "pytest transcript"):
            _verify_pytest_transcript(process.stdout + "\nplugin says 999 passed\n", self.directory / "junit.xml")

        tree = ET.parse(self.directory / "junit.xml")
        suite = tree.getroot().find("testsuite")
        assert suite is not None
        subtest = next(case for case in suite.findall("testcase") if "illm-subtest#" in case.attrib["name"])
        subtest.set("name", subtest.attrib["name"].replace("illm-subtest#1:", "illm-subtest#9:", 1))
        tampered = self.directory / "tampered-junit.xml"
        tree.write(tampered, encoding="utf-8")
        with self.assertRaisesRegex(IllmError, "reordered, duplicated, or detached"):
            _verify_pytest_transcript(process.stdout, tampered)

    def test_skipped_subtest_is_visible_in_junit_and_fails_closed(self) -> None:
        process = self._run('''import pytest
def test_vectors(subtests):
    with subtests.test("skipped child", value=1):
        pytest.skip("actual-child-skip")
    with subtests.test("passing child", value=2):
        assert True
''', quiet=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertRegex(process.stdout, r"(?m)^=+ 1 passed in [0-9.]+s =+$")
        self.assertEqual(parse_junit(self.directory / "junit.xml")["skipped"], 1)
        with self.assertRaisesRegex(IllmError, "skipped or xfailed subtests"):
            _verify_pytest_transcript(process.stdout, self.directory / "junit.xml")

    def test_ordinary_parent_failure_remains_a_failure(self) -> None:
        process = self._run('''import unittest
class Vectors(unittest.TestCase):
    def test_vectors(self):
        with self.subTest(value=1):
            self.assertTrue(True)
        self.fail("actual-parent-failure")
''')
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        suite, cases = self._suite()
        self.assertEqual(len(cases), 2)
        self.assertEqual(int(suite.attrib["failures"]), 1)
        parent = next(case for case in cases if case.attrib["name"] == "test_vectors")
        self.assertIn("actual-parent-failure", parent.findtext("failure") or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
