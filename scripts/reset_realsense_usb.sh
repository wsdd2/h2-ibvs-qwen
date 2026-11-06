#!/bin/bash
# Release RealSense USB ports stuck after an unclean camera exit.
# Does not reboot the Orin. Needs root.
set -u
found=0
for dev in /sys/bus/usb/devices/*; do
  [ -f "${dev}/idVendor" ] || continue
  vendor="$(cat "${dev}/idVendor" 2>/dev/null || true)"
  [ "${vendor}" = "8086" ] || continue
  name="$(basename "${dev}")"
  product="$(cat "${dev}/idProduct" 2>/dev/null || true)"
  echo "reset ${name} ${vendor}:${product}"
  if [ -w /sys/bus/usb/drivers/usb/unbind ]; then
    echo "${name}" > /sys/bus/usb/drivers/usb/unbind || true
    sleep 1
    echo "${name}" > /sys/bus/usb/drivers/usb/bind || true
  elif [ -w "${dev}/authorized" ]; then
    echo 0 > "${dev}/authorized" || true
    sleep 1
    echo 1 > "${dev}/authorized" || true
  else
    echo "cannot write ${dev}; run with sudo" >&2
    exit 1
  fi
  found=1
done
if [ "${found}" -eq 0 ]; then
  echo "no Intel 8086 device in sysfs"
  echo "the USB port is stuck. A cable replug did not reset the Orin controller."
  exit 1
fi
sleep 1
echo "done. run: rs-enumerate-devices -s"
