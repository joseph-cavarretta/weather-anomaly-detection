#!/bin/bash
docker build -t weather-model "$(git rev-parse --show-toplevel)"
