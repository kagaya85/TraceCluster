#!/usr/bin/env bash

read -r -p "Remove processed pt file? [Y/n] " input

case $input in
	[yY][eE][sS]|[yY])
		rm -rf data/processed/*
		;;
	*)
		echo "Skip"
		;;

read -r -p "Remove weights file? [Y/n] " input 

case $input in
	[yY][eE][sS]|[yY])
		rm -rf weights/*
		;;
	*)
		echo "Skip"
		;;
