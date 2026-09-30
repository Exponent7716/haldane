#!/bin/sh
# Sweep load capacitance and print inverter delays.
cd "$(dirname "$0")"
printf "%-6s %-12s %-12s %-12s %-12s %-12s\n" CL tphl tplh tpd trise tfall
for cl in 5f 10f 20f 40f 80f; do
  sed "s/CL=20f/CL=$cl/" inverter.cir > /tmp/inv_$$.cir
  ngspice -b /tmp/inv_$$.cir 2>&1 | awk -v cl=$cl '
    /^tphl /{a=$3} /^tplh /{b=$3} /^tpd /{c=$3} /^trise /{d=$3} /^tfall /{e=$3}
    END{printf "%-6s %-12s %-12s %-12s %-12s %-12s\n", cl,a,b,c,d,e}'
done
rm -f /tmp/inv_$$.cir
