#!/bin/bash
function usage {
  # expXXX or XXX is expected as the argument
  cat <<EOF
$(basename ${0}) is a wrapper to execute experiment files

Usage:
  $(basename ${0}) [command] [<options>]

Command:
  experiment number to execute (e.g. exp001 or 001)

Options:
  --help, -h    print this
EOF
}

for OPT in "$@"; do
    case "$OPT" in
    '-h' | '--help')
        usage
        exit 1
        ;;
    esac
done

EXP=$1
if [ $EXP == "" ]; then
  echo "Need experiment number to run"
  exit 1
fi
if [ ! "`echo $EXP | grep 'exp'`" ]; then
  EXP="exp$EXP"
fi
echo "EXP: ${EXP}"
python -m src.run -e $EXP
