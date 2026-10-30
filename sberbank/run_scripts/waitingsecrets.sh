#!/bin/sh

config_file=/tmp/secman/config/secman.txt
checked_files=`cat $config_file`
files_count=0
for file in $checked_files ; do
    files_count=$(( files_count + 1 ))
done
exists_files_count=0
time_counter=0
while [ $exists_files_count != $files_count ]; do
    exists_files_count=0
    for file in $checked_files ; do
        if [ -f $file ]; then
            exists_files_count=$(( exists_files_count + 1 ))
        fi
    done
    sleep 1
    time_counter=$(( time_counter + 1 ))
    echo "Waiting $time_counter s."
done
