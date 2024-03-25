PostgreSQL is not included into the Docker image but instead must be set up
externally. Here are the instructions for setting up PostgreSQL on a Linux host.


1. Set up repository with PostgreSQL extensions:

```shell
sudo apt install -y postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
```

2. Install PostgreSQL and pgvector:

```shell
sudo apt update
sudo apt install -y postgresql postgresql-14-pgvector
```

3. (Re)start PostgreSQL

```shell
sudo service postgresql restart
```

4. Create user and database, enable pgvector:

```shell
# start psql
sudo -u postgres psql
```

the in the psql:

```sql
-- change values for production
\set username kava
\set password kava
\set dbname kava

CREATE USER :username WITH PASSWORD :'password';
CREATE DATABASE :dbname;
GRANT ALL PRIVILEGES ON DATABASE :dbname TO :username;

\c :dbname

CREATE EXTENSION vector;

-- exit psql
```

5. Check that you can connect to the databsase from the shell:

```shell
psql -W -U kava -h 127.0.0.1 kava
# enter the password specified above
```

PostgreSQL client is already installed in the Docker image, and network is
set to host, so you should be able to connect to the database from container too.