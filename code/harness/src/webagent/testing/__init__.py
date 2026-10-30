"""Offline test doubles. Not imported by production modules."""

from .fake_backend import FakeBackend, FakeElement, FakePage, FakeSite

__all__ = ["FakeBackend", "FakeElement", "FakePage", "FakeSite"]
