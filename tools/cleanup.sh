#!/usr/bin/env bash

ProcessedPath='data/processed/'
WeightsPath='weights/*'

read -r -p "Remove processed pt file? ($ProcessedPath) [Y/n] " input

case $input in
	[yY][eE][sS]|[yY])
		rm -rf $ProcessedPath
		mkdir $ProcessedPath 
		echo "Removed"
		;;
	*)
		echo "Skip"
		;;
esac

read -r -p "Remove weights file? ($WeightsPath) [Y/n] " input 

case $input in
	[yY][eE][sS]|[yY])
		rm -rf $WeightsPath
		echo "Removed"
		;;
	*)
		echo "Skip"
		;;
esac