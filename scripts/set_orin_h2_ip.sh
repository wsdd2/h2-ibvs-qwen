#!/bin/bash
# Keep the Orin link to the H2 on a static address.
# NetworkManager was waiting on DHCP and deleting 192.168.123.166.
# Does not change the Wi-Fi default route.
set -euo pipefail

IFACE="${1:-eth2}"
ADDRESS="${2:-192.168.123.166/24}"
PEER="${3:-192.168.123.162}"

if ! command -v nmcli >/dev/null 2>&1; then
  echo "nmcli is required" >&2
  exit 1
fi

name="$(nmcli -g GENERAL.CONNECTION device show "$IFACE" 2>/dev/null || true)"
if [ -z "$name" ] || [ "$name" = "--" ]; then
  name="$(nmcli -t -f NAME,DEVICE connection show | awk -F: -v dev="$IFACE" '$2==dev {print $1; exit}')"
fi
if [ -z "$name" ] || [ "$name" = "--" ]; then
  name="orin-h2"
  echo "creating NetworkManager connection ${name} on ${IFACE}"
  sudo nmcli connection add type ethernet ifname "$IFACE" con-name "$name" \
    ipv4.method manual \
    ipv4.addresses "$ADDRESS" \
    ipv4.never-default yes \
    ipv6.method disabled \
    connection.autoconnect yes
else
  echo "updating ${name} on ${IFACE}"
  sudo nmcli connection modify "$name" \
    connection.interface-name "$IFACE" \
    ipv4.method manual \
    ipv4.addresses "$ADDRESS" \
    ipv4.never-default yes \
    ipv6.method disabled \
    connection.autoconnect yes
fi

sudo nmcli connection up "$name"
echo "=== ${IFACE} ==="
ip -4 addr show "$IFACE"
echo "=== ping ${PEER} ==="
ping -c 2 -I "$IFACE" "$PEER"
