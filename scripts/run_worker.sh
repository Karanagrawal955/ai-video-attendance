#!/usr/bin/env bash
# Celery worker entrypoint: one dedicated queue per camera slot + control/jobs.
#
#   CAMERA_SLOTS=8 -> queues: control,jobs,camera_slot_0..camera_slot_7
#   concurrency    -> CAMERA_SLOTS + 2 (the +2 keeps control/jobs responsive
#                     while every slot is occupied by a long camera task)
set -euo pipefail

SLOTS="${CAMERA_SLOTS:-8}"
QUEUES="control,jobs"
for ((i = 0; i < SLOTS; i++)); do
  QUEUES="${QUEUES},camera_slot_${i}"
done
CONCURRENCY=$((SLOTS + 2))

echo "celery worker | queues=${QUEUES} | concurrency=${CONCURRENCY}"

exec celery -A app.workers.celery_app worker \
  -Q "${QUEUES}" \
  --concurrency="${CONCURRENCY}" \
  --prefetch-multiplier=1 \
  --loglevel="${CELERY_LOGLEVEL:-info}" \
  --hostname="worker@%h"
