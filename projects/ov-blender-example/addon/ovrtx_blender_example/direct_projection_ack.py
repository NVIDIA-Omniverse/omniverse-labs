# SPDX-License-Identifier: Apache-2.0
"""Small native acknowledgment index over the coordinator's current projection.

No payload copies and no edit history. Attribute keys retain one current version;
no-op selection is O(1). New populations replay current values, camera/value
writes do not invalidate the token. Caller supplies a proven population token.
"""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ReplayTicket:
    token: object
    rows: tuple
    versions: tuple
    full_replay: bool

@dataclass
class ProjectionAck:
    accepted_token: object = None
    versions: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    serial: int = 0

    def changed(self, key):
        self.serial += 1
        self.versions[key] = self.serial
        self.pending[key] = self.serial

    def select(self, current, token):
        full = token is None or token != self.accepted_token
        # Index only changed keys on a warm population; never inspect large
        # arrays to rediscover whether a compiled value is unchanged.
        keys = current.keys() if full else self.pending.keys()
        selected = tuple((key, self.versions.get(key, 0)) for key in keys if key in current)
        return ReplayTicket(token, tuple(current[key] for key, _ in selected), selected, full)

    def acknowledge(self, ticket, token_after):
        """Called ONLY after complete native acceptance; false means replay again.

        A worker/population replacement during submission cannot certify skipped
        keys. A newer compilation of the same key remains pending after old ACK.
        """
        if ticket.token is None or token_after != ticket.token:
            self.accepted_token = None
            return False
        self.accepted_token = token_after
        for key, version in ticket.versions:
            if self.pending.get(key) == version:
                self.pending.pop(key, None)
        return True

    def forget(self, keys):
        """Population owner removes retired projection keys, not every capture."""
        for key in keys:
            self.versions.pop(key, None)
            self.pending.pop(key, None)

    def invalidate(self):
        self.accepted_token = None
