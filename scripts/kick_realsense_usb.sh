#!/bin/bash
# Bring RealSense devices back without rebooting the Orin and without
# unbinding tegra-xusb. Port power cycle first. If this kernel has no
# port disable nodes, only Intel 8086 devices are authorized-cycled.
set -u
n=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  echo "off $f"
  echo 1 > "$f" 2>/dev/null || true
  n=$((n + 1))
done < <(find /sys/bus/usb/devices -name disable 2>/dev/null)
sleep 2
while IFS= read -r f; do
  [ -n "$f" ] || continue
  echo "on $f"
  echo 0 > "$f" 2>/dev/null || true
done < <(find /sys/bus/usb/devices -name disable 2>/dev/null)
if [ "$n" -eq 0 ]; then
  echo "no port disable nodes; authorized-cycle Intel 8086"
  for d in /sys/bus/usb/devices/[0-9]*; do
    [ -f "$d/idVendor" ] || continue
    [ "$(cat "$d/idVendor" 2>/dev/null)" = "8086" ] || continue
    echo "reset $(basename "$d") serial=$(cat "$d/serial" 2>/dev/null)"
    echo 0 > "$d/authorized" 2>/dev/null || true
    sleep 1
    echo 1 > "$d/authorized" 2>/dev/null || true
  done
fi
sleep 2
echo "keep Intel cameras awake"
for d in /sys/bus/usb/devices/[0-9]*; do
  [ -f "$d/idVendor" ] || continue
  [ "$(cat "$d/idVendor" 2>/dev/null)" = "8086" ] || continue
  echo on > "$d/power/control" 2>/dev/null || true
  echo -1 > "$d/power/autosuspend" 2>/dev/null || true
  echo -1 > "$d/power/autosuspend_delay_ms" 2>/dev/null || true
done
echo "kick done"
