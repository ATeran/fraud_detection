#!/bin/bash 


# check for existence of the database and
# create it if it doesn't exist

if psql -lqt | cut -d \| -f 1 | grep -qw frauddb; then
    # database exists
    # $? is 0
    echo "database exists";
else    
    sudo -u postgres psql -c create database frauddb
    sudo -u postrges psql -f create_table.sql
fi
