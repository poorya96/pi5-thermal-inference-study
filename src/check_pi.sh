#!/bin/bash
echo "=== Pi Status Check ==="
echo "Time:        $(date)"
echo "Temperature: $(vcgencmd measure_temp)"
echo "CPU Clock:   $(vcgencmd measure_clock arm)"
echo "Throttle:    $(vcgencmd get_throttled)"
echo "RAM Free:    $(free -h | awk '/^Mem:/ {print $4}')"
echo "CPU Load:    $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}')% used"
echo "======================="
