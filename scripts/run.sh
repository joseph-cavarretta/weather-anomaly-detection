#!/bin/bash
docker run \
--rm \
--volume ~/projects/weather_model/src/data:/data \
--volume ~/projects/weather_model/src/data/scheduled_runs:/output \
weather_model