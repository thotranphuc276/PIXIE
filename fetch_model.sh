#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
username="thochit2762003@gmail.com"
password="tho2762003"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# PIXIE pretrained model and utilities
echo -e "\nYou need to register at https://pixie.is.tue.mpg.de/"
username="thochit2762003@gmail.com"
password="tho2762003"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './data/pixie_model.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './data/utilities.zip' --no-check-certificate --continue

sudo apt-get update
sudo apt-get install unzip

# shellcheck disable=SC2164
cd ./data
unzip -o -q utilities.zip
