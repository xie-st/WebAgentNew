"""Coding domain prompt templates."""

from .bug_diagnose_v1 import bug_diagnose_v1
from .code_review_v1 import code_review_v1
from .docstring_fill_v1 import docstring_fill_v1
from .refactor_plan_v1 import refactor_plan_v1
from .test_generate_v1 import test_generate_v1

__all__ = [
    "code_review_v1",
    "bug_diagnose_v1",
    "refactor_plan_v1",
    "test_generate_v1",
    "docstring_fill_v1",
]
