#!/bin/bash
set -e

mypy . --install-types --non-interactive
isort .
black .