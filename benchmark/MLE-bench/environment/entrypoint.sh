#!/bin/bash

# Print commands and their arguments as they are executed
set -x

{
  # log into /home/logs
  LOGS_DIR=/home/logs
  mkdir -p $LOGS_DIR

  # REQUIRED: chmod so agent (nonroot or host user) can read/write and traverse dirs.
  # Without this, agent cannot create /home/logs/subagent_logs, /home/code, /home/submission, etc.
  # a+rwX: capital X = add execute only for dirs, so agent can cd and create subdirs.
  # -path /home/data -prune excludes the read-only competition data volume.
  find /home -path /home/data -prune -o -exec chmod a+rwX {} \;
  ls -l /home

  # Launch grading server, stays alive throughout container lifetime to service agent requests.
  /opt/conda/bin/conda run -n mleb python /private/grading_server.py
} 2>&1 | tee $LOGS_DIR/entrypoint.log
