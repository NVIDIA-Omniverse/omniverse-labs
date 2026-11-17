#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral and structural contracts for the GitHub CI workflow."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ACTION_REFS = {
    "actions/checkout": (
        "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "v7",
    ),
    "actions/create-github-app-token": (
        "bcd2ba49218906704ab6c1aa796996da409d3eb1",
        "v3",
    ),
    "actions/setup-python": (
        "ece7cb06caefa5fff74198d8649806c4678c61a1",
        "v6",
    ),
    "actions/upload-artifact": (
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "v7",
    ),
    "actions/download-artifact": (
        "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "v8",
    ),
}


def pinned_action(name: str) -> str:
    commit, version = ACTION_REFS[name]
    return f"{name}@{commit} # {version}"


def fail(message: str) -> None:
    raise AssertionError(f"pipeline structure failure: {message}")


def parse_named_blocks(
    lines: list[str], start_header: str, entry_pattern: str
) -> dict[str, list[str]]:
    try:
        start = lines.index(start_header) + 1
    except ValueError:
        fail(f"missing {start_header}")
    entry_re = re.compile(entry_pattern)
    entries: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start:]:
        match = entry_re.fullmatch(line)
        if match:
            current = match.group(1)
            if current in entries:
                fail(f"duplicate entry {current}")
            entries[current] = [line]
        elif current is not None:
            if line and not line.startswith(" "):
                break
            entries[current].append(line)
    return entries


def scalar(block: list[str], indent: int, key: str) -> str:
    prefix = " " * indent + key + ":"
    matches = [line[len(prefix) :].strip() for line in block if line.startswith(prefix)]
    if len(matches) != 1:
        fail(f"expected one {key} field, found {len(matches)}")
    return matches[0]


def optional_scalar(block: list[str], indent: int, key: str) -> str:
    prefix = " " * indent + key + ":"
    matches = [line[len(prefix) :].strip() for line in block if line.startswith(prefix)]
    if len(matches) > 1:
        fail(f"expected at most one {key} field, found {len(matches)}")
    return matches[0] if matches else ""


def list_value(block: list[str], indent: int, key: str) -> list[str]:
    header = " " * indent + key + ":"
    try:
        start = block.index(header) + 1
    except ValueError:
        fail(f"missing list {key}")
    item_prefix = " " * (indent + 2) + "- "
    values: list[str] = []
    for line in block[start:]:
        if line.startswith(item_prefix):
            values.append(line[len(item_prefix) :].strip())
        elif line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
    return values


def step_blocks(job: list[str]) -> list[list[str]]:
    try:
        start = job.index("    steps:") + 1
    except ValueError:
        fail("job has no steps")
    steps: list[list[str]] = []
    current: list[str] | None = None
    for line in job[start:]:
        if line.startswith("      - name: "):
            current = [line]
            steps.append(current)
        elif current is not None:
            if line.strip() and len(line) - len(line.lstrip()) <= 4:
                break
            current.append(line)
    return steps


def step_name(step: list[str]) -> str:
    return step[0][len("      - name: ") :]


def nested_map(block: list[str], indent: int, key: str) -> dict[str, str]:
    header = " " * indent + key + ":"
    try:
        start = block.index(header) + 1
    except ValueError:
        return {}
    child_indent = indent + 2
    result: dict[str, str] = {}
    entry_re = re.compile(rf"^ {{{child_indent}}}([A-Za-z0-9_-]+):\s*(.*)$")
    for line in block[start:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        match = entry_re.fullmatch(line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


TOKEN_RE = re.compile(
    r"\s*(&&|\|\||==|!=|\(|\)|'(?:[^']*)'|[A-Za-z_][A-Za-z0-9_.-]*)"
)


class ExpressionParser:
    def __init__(self, expression: str, context: dict[str, str]) -> None:
        self.tokens = self._tokenize(expression)
        self.context = context
        self.index = 0

    @staticmethod
    def _tokenize(expression: str) -> list[str]:
        tokens: list[str] = []
        offset = 0
        while offset < len(expression):
            match = TOKEN_RE.match(expression, offset)
            if not match:
                fail(f"unsupported workflow condition near {expression[offset:]!r}")
            tokens.append(match.group(1))
            offset = match.end()
        return tokens

    def parse(self) -> bool:
        result = self.parse_or()
        if self.index != len(self.tokens):
            fail(f"unexpected condition token {self.tokens[self.index]!r}")
        return bool(result)

    def parse_or(self) -> bool:
        result = self.parse_and()
        while self.take("||"):
            right = self.parse_and()
            result = bool(result or right)
        return bool(result)

    def parse_and(self) -> bool:
        result = self.parse_comparison()
        while self.take("&&"):
            right = self.parse_comparison()
            result = bool(result and right)
        return bool(result)

    def parse_comparison(self) -> bool | str:
        left = self.parse_primary()
        if self.take("=="):
            return left == self.parse_primary()
        if self.take("!="):
            return left != self.parse_primary()
        return left

    def parse_primary(self) -> bool | str:
        if self.take("("):
            result = self.parse_or()
            if not self.take(")"):
                fail("unclosed condition parenthesis")
            return result
        if self.index >= len(self.tokens):
            fail("unexpected end of workflow condition")
        token = self.tokens[self.index]
        self.index += 1
        if token.startswith("'"):
            return token[1:-1]
        if token in {"true", "false"}:
            return token == "true"
        return self.context.get(token, "")

    def take(self, token: str) -> bool:
        if self.index < len(self.tokens) and self.tokens[self.index] == token:
            self.index += 1
            return True
        return False


def evaluate(expression: str, **context: str) -> bool:
    if expression.startswith("${{") and expression.endswith("}}"):
        expression = expression[3:-2].strip()
    return ExpressionParser(expression, context).parse()


def render_concurrency(template: str, **context: str) -> str:
    match = re.fullmatch(
        r"(.*)\$\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\|\|\s*"
        r"([A-Za-z_][A-Za-z0-9_.-]*)\s*}}(.*)",
        template,
    )
    if not match:
        fail("concurrency group must use a parsed PR-number/run-id fallback")
    value = context.get(match.group(2), "") or context.get(match.group(3), "")
    return f"{match.group(1)}{value}{match.group(4)}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workflow_path = repo_root / ".github" / "workflows" / "ci.yml"
    gitlab_path = repo_root / ".gitlab-ci.yml"
    lines = workflow_path.read_text(encoding="utf-8").splitlines()
    jobs = parse_named_blocks(lines, "jobs:", r"^  ([a-z0-9_-]+):$")

    action_counts = {name: 0 for name in ACTION_REFS}
    for job_name, job in jobs.items():
        for step in step_blocks(job):
            action = optional_scalar(step, 8, "uses")
            if not action:
                continue
            matches = [name for name in ACTION_REFS if action == pinned_action(name)]
            if len(matches) != 1:
                fail(f"{job_name} contains an unverified action reference: {action}")
            action_counts[matches[0]] += 1
    expected_action_counts = {
        "actions/checkout": 8,
        "actions/create-github-app-token": 4,
        "actions/setup-python": 2,
        "actions/upload-artifact": 4,
        "actions/download-artifact": 0,
    }
    if action_counts != expected_action_counts:
        fail(f"workflow action counts differ from the preserved wiring: {action_counts}")

    expected_producers = {
        "verify-root-windows": "verify-root-windows-${{ github.sha }}",
        "verify-root-linux": "verify-root-linux-${{ github.sha }}",
        "verify-ovdnext-data-windows": "verify-ovdnext-data-windows-${{ github.sha }}",
        "verify-ovdnext-data-linux": "verify-ovdnext-data-linux-${{ github.sha }}",
    }
    producers: dict[str, str] = {}
    for job_name, expected_name in expected_producers.items():
        upload_steps = [
            step
            for step in step_blocks(jobs[job_name])
            if optional_scalar(step, 8, "uses") == pinned_action("actions/upload-artifact")
        ]
        if len(upload_steps) != 1:
            fail(f"{job_name} must have exactly one artifact producer")
        inputs = nested_map(upload_steps[0], 8, "with")
        if inputs.get("name") != expected_name:
            fail(f"{job_name} changed its producer artifact name")
        if inputs.get("overwrite") != "true":
            fail(f"{job_name} artifact producer must enable overwrite")
        if expected_name in producers:
            fail(f"producer artifact name is not unique: {expected_name}")
        producers[expected_name] = job_name
    if len(producers) != 4:
        fail(f"expected four unique artifact producers, found {len(producers)}")

    verification_jobs = {
        "verify-root-windows", "verify-root-linux",
        "verify-ovdnext-data-windows", "verify-ovdnext-data-linux",
    }
    if set(jobs) != verification_jobs:
        fail("root workflow must contain only the four preserved native verification jobs")
    if scalar(lines, 2, "contents") != "read":
        fail("root verification must retain read-only repository permissions")
    modern = (repo_root / ".github/workflows/ovui.yml").read_text(encoding="utf-8")
    for forbidden in ("qtbase5", "qwt-qt5", "aqtinstall", "reference-qt-root", "install_verified_qt"):
        if forbidden in "\n".join(lines) or forbidden in modern:
            fail(f"obsolete toolkit provisioning remains: {forbidden}")
    for required in ("verify-package:", "publish:", "ci/verify_ovui_package.py", "ci/release_ovui.py"):
        if required not in modern:
            fail(f"modern package/publication gate missing: {required}")

    concurrency_lines = parse_named_blocks(
        lines, "concurrency:", r"^  ([a-z-]+):\s*(.*)$"
    )
    concurrency_group = scalar(
        [entry[0] for entry in concurrency_lines.values()], 2, "group"
    )
    pr_one = render_concurrency(
        concurrency_group,
        **{"github.event.pull_request.number": "42", "github.run_id": "1001"},
    )
    pr_two = render_concurrency(
        concurrency_group,
        **{"github.event.pull_request.number": "42", "github.run_id": "1002"},
    )
    push_one = render_concurrency(
        concurrency_group,
        **{"github.event.pull_request.number": "", "github.run_id": "1001"},
    )
    push_two = render_concurrency(
        concurrency_group,
        **{"github.event.pull_request.number": "", "github.run_id": "1002"},
    )
    if pr_one != pr_two:
        fail("pull-request runs must share a concurrency group for supersession")
    if push_one == push_two:
        fail("non-pull-request runs must have unique concurrency groups")

    token_route = "${{ steps.physics-token.outputs.token || secrets.OVDNEXT_CI_PHYSICS_TOKEN || github.token }}"
    for job_name in verification_jobs:
        job = jobs[job_name]
        if nested_map(job, 4, "env").get("OVDNEXT_CI_APP_ID") != "${{ secrets.OVDNEXT_CI_APP_ID }}":
            fail(f"{job_name} changed the optional App ID route")
        by_name = {step_name(step): step for step in step_blocks(job)}
        app_inputs = nested_map(by_name["Create Physics checkout token"], 8, "with")
        if app_inputs.get("client-id") != "${{ env.OVDNEXT_CI_APP_ID }}" or app_inputs.get(
            "private-key"
        ) != "${{ secrets.OVDNEXT_CI_APP_PRIVATE_KEY }}":
            fail(f"{job_name} changed the optional App credential route")
        checkout_inputs = nested_map(by_name["Check out Physics"], 8, "with")
        if checkout_inputs.get("token") != token_route:
            fail(f"{job_name} changed the approved Physics credential fallback route")

    gitlab_text = gitlab_path.read_text(encoding="utf-8")
    significant_gitlab = [
        line for line in gitlab_text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if significant_gitlab != ["workflow:", "  rules:", "    - when: never"]:
        fail("GitLab configuration must be an unconditional workflow tombstone")
    if ".github/workflows/ci.yml" not in gitlab_text:
        fail("GitLab tombstone must point readers to the GitHub workflow")

    print("PASS: parsed GitHub pipeline relationships and GitLab tombstone are enforced")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, StopIteration, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
