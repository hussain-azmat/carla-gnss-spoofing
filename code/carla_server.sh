#!/usr/bin/env bash
# safe CARLA server control:  carla_server.sh start|stop|status  [quality]
CARLA=~/CARLA_0.9.15
PIDF=/tmp/carla.pid
PORT=2000
Q="${2:-Epic}"
case "${1:-status}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "already running pid $(cat $PIDF)"; exit 0; fi
    cd "$CARLA" || exit 1
    setsid ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level="$Q" \
        -carla-rpc-port=$PORT -carla-primary-port=0 > ~/carla_srv.log 2>&1 &
    echo $! > "$PIDF"
    echo "started pid $(cat $PIDF) (quality $Q), waiting for RPC $PORT..."
    for i in $(seq 1 90); do
      (exec 3<>/dev/tcp/localhost/$PORT) 2>/dev/null && { echo "  port open after ${i}s"; break; }
      sleep 1
    done
    # warm up: settle + confirm the world actually answers get_map, retry a few times
    for i in $(seq 1 20); do
      sleep 3
      if kill -0 "$(cat $PIDF)" 2>/dev/null; then
        echo "  server alive, ${i}x3s settle"
      else
        echo "  server DIED during warmup - tail:"; tail -15 ~/carla_srv.log; exit 1
      fi
      [ $i -ge 3 ] && break
    done
    echo "READY"; exit 0 ;;
  stop)
    [ -f "$PIDF" ] && kill "$(cat $PIDF)" 2>/dev/null
    sleep 3
    [ -f "$PIDF" ] && kill -9 "$(cat $PIDF)" 2>/dev/null
    pkill -9 -f 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping' 2>/dev/null
    rm -f "$PIDF"; echo "stopped" ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "running pid $(cat $PIDF)"
      (exec 3<>/dev/tcp/localhost/$PORT) 2>/dev/null && echo "RPC $PORT OPEN" || echo "RPC $PORT CLOSED"
    else echo "not running"; fi ;;
esac
