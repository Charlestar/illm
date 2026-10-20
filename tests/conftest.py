"""Keep every executed subtest as an independent, truthful JUnit testcase.

The locked pytest 9.1.1 reporter increments suite counters for SubtestReport,
but merges reports sharing their parent nodeid into one testcase.  This adapter
only gives those reports a temporary XML identity; execution and outcomes still
come from pytest, and the strict release-side JUnit reader remains unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from _pytest.junitxml import LogXML, xml_key
from _pytest.reports import TestReport
from _pytest.subtests import SubtestReport


class _IndependentSubtestJUnitCases:
    def __init__(self, xml: LogXML) -> None:
        self.xml = xml
        self.sequence = 0

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_logreport(self, report: TestReport) -> Iterator[None]:
        if not isinstance(report, SubtestReport):
            yield
            return

        self.sequence += 1
        original_nodeid = report.nodeid
        context = json.dumps(report.context._to_json(), ensure_ascii=True, sort_keys=True)
        report.nodeid = f"{original_nodeid}[illm-subtest#{self.sequence}:{context}]"
        try:
            yield
        finally:
            try:
                # A subtest has a call report, not its own teardown report.
                # Finalize exactly the case written by the built-in reporter.
                self.xml.finalize(report)
                self.xml.open_reports[:] = [item for item in self.xml.open_reports if item is not report]
            finally:
                report.nodeid = original_nodeid


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    xml = config.stash.get(xml_key, None)
    if xml is None:
        return
    if pytest.__version__ != "9.1.1":
        raise pytest.UsageError("independent subtest JUnit adapter requires locked pytest 9.1.1")
    config.pluginmanager.register(_IndependentSubtestJUnitCases(xml), "illm-independent-subtest-junit")
