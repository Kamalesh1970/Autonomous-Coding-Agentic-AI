#!/usr/bin/env bash
cd '/home/kamalesh/Autonomous Coding Agentic AI'
source .venv/bin/activate
python -m pytest -v --tb=short tests/test_delivery.py -k "plan_approval" > /tmp/pytest_plan_approval.txt 2>&1
echo "EXIT:$?" >> /tmp/pytest_plan_approval.txt
python -m pytest -v --tb=short > /tmp/pytest_full.txt 2>&1
echo "FULL_EXIT:$?" >> /tmp/pytest_full.txt
