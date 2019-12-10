#!/bin/bash
while true;
do
	ps cax | grep torcs > /dev/null
	if [ $? -eq 0 ]; then
	  : #echo "Process is running."
	else
	  echo "Process is not running."
	  torcs & sh autostart.sh      	  	  
	fi
done;


# Will be soon deprecated (No longer supported by MADRaS)