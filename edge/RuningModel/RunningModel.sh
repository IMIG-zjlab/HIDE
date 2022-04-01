#!/usr/bin/env bash

python3 app_mt.py
python3 -m vaitrace_py ./app_mt.py
xmutil platformstats -p

