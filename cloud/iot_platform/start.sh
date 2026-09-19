#!/usr/bin/env bash
set -e

if [ -z "$CANDIDATE" ]; then
  CANDIDATE=$(hostname -I | awk '{print $1}')
  echo "未检测到 CANDIDATE，自动使用本机IP: $CANDIDATE"
fi

SRS_NAME="nightowl-srs"

cleanup() {
  echo
  echo "正在停止服务..."

  if [ -n "$PY_PID" ] && kill -0 "$PY_PID" 2>/dev/null; then
    kill "$PY_PID" 2>/dev/null || true
  fi

  docker stop "$SRS_NAME" >/dev/null 2>&1 || true

  wait 2>/dev/null || true
  echo "服务已停止"
}

trap cleanup INT TERM EXIT

echo "清理旧容器..."
 docker rm -f "$SRS_NAME" >/dev/null 2>&1 || true

echo "启动 SRS..."
docker run --rm -d \
  --name "$SRS_NAME" \
  -e CANDIDATE="$CANDIDATE" \
  -p  \
  -p  \
  -p  \
  -p  \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:6 \
  objs/srs -c conf/rtmp2rtc.conf >/dev/null

sleep 3

echo "启动 websocket1v3.py..."
python3 websocket1v4.py &
PY_PID=$!

echo "CANDIDATE: $CANDIDATE"
echo "SRS 容器: $SRS_NAME"
echo "websocket.py PID: $PY_PID"
echo "按 Ctrl+C 停止所有服务"

wait "$PY_PID"
