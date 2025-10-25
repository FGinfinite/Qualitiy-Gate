#!/bin/bash
# Description: This script runs the second stage of the project: data selection.
# Usage: ./scripts/run_stage_2.sh

.venv/bin/python src/main.py --config-name=stage_2_selection "$@"