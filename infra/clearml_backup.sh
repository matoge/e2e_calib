#!/bin/bash
# ClearML files_server → SMB backup
# Mirrors /DATADISK2/clearml_server/data/fileserver → /mnt/ecp-perception/clearml_backup/fileserver
# Runs nightly at 02:00 via cron (heatrun host).
# Logs to /var/log/clearml_backup.log (rotation handled by logrotate.d).

SRC=/DATADISK2/clearml_server/data/fileserver/
DST=/mnt/ecp-perception/clearml_backup/fileserver/
LOG=/var/log/clearml_backup.log

mkdir -p "$DST"
exec >> "$LOG" 2>&1
echo "[$(date -Iseconds)] start"
# -a: archive mode (perms, times, symlinks)
# --delete: remove dst files absent in src (clean mirror)
# --info=stats2: throughput summary
# --exclude .lock, .tmp: skip transient files
rsync -a --delete --info=stats2 \
    --exclude='*.lock' --exclude='*.tmp' --exclude='.in_progress' \
    "$SRC" "$DST"
echo "[$(date -Iseconds)] done rc=$?"
