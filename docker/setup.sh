#!/bin/bash

export NAME=bliq

# Install PostgreSQL and pgvector

apt install -y postgresql-common
/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y

apt update
apt install -y postgresql postgresql-14-pgvector
service postgresql restart

# Create a dev user

sed 's/{NAME}/bliq/p' setup_pg_template.sql

su postgres -c "psql --file=setup_pg.sql"

su postgres -c "psql -c \"CREATE USER ${NAME} WITH PASSWORD '${NAME}';\""
su postgres -c "psql -c \"CREATE DATABASE ${NAME};\""
su postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE ${NAME} TO ${NAME};\""
su postgres -c "psql -c \"CREATE EXTENSION vector;\""

