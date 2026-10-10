#!/bin/sh
set -e

mkdir -p /config /app/data /app/logs

# The app reads/writes ./config.yaml; keep the real file in the /config volume
[ -f /config/config.yaml ] || cp /app/config.yaml.example /config/config.yaml
ln -sf /config/config.yaml /app/config.yaml

exec "$@"
