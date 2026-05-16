#!/bin/bash
ROOT="$(git rev-parse --show-toplevel)"
docker run \
  --rm \
  --volume "${ROOT}/src/data:/data" \
  --volume "${ROOT}/src/data/scheduled_runs:/output" \
  weather-model
