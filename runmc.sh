# input files required for running memc2
# clusters.out
# eci.out
# gs_str.out
# lat.in
# str.out
# control.in

# output files written
# mc.out         : energy is written on second row 23rd column?
# mcheader.out   : read more information
# mcsnapshot.out : structure is written

SECONDS=0
memc2 -is=str.out -n=0 -eq=0 -g2c -q -sigdig=10
tail -1 mc.out | cut -f24
ENDTIME=$(date +%s)
echo $SECONDS
