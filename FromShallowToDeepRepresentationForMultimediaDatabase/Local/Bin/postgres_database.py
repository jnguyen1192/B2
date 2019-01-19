#!/usr/bin/python
import psycopg2
from config import config
import numpy as np
#http://www.postgresqltutorial.com/postgresql-python/connect/


def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def insert_image_with_descriptor(name_image, des):
    """ insert image into the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        cur.execute("INSERT INTO image (name_image, des) VALUES (%s, %s)", (name_image, des))
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


# converts from python to postgres
def _adapt_array(text):
    from tempfile import TemporaryFile
    outfile = TemporaryFile()
    np.savetxt(outfile, text)
    outfile.seek(0)
    return str(outfile.read())[2:-1].replace("\\n", "\n")


# converts from postgres to python
def _typecast_array(string):
    from tempfile import TemporaryFile
    outfile = TemporaryFile()
    outfile.write(string.encode())
    outfile.seek(0)  # Only needed here to simulate closing & reopening file
    return np.loadtxt(outfile, dtype=np.float32)


def insert_cluster_with_descriptor(num_cluster, des):
    """ insert image into the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()
        #print(_adapt_array(des)[:10])
        # execute a statement
        cur.execute("INSERT INTO cluster(num_cluster, des) VALUES (%s, %s)", (num_cluster, _adapt_array(des)))
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def insert_has_cluster_with_num_cluster_and_name_image(num_cluster, name_image):
    """ insert image into the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        cur.execute("INSERT INTO has_cluster(num_cluster, name_image) VALUES (%s, %s)", (num_cluster, name_image))
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def get_number_cluster():
    """ insert image into the PostgreSQL database server """
    conn = None
    result = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        cur.execute("SELECT count(*) FROM cluster")

        result = cur.fetchone()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return result[0]


def get_descriptor_cluster_with_num_cluster(num_cluster):
    """ insert image into the PostgreSQL database server """
    conn = None
    result = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        cur.execute("SELECT des FROM cluster WHERE num_cluster=%s", (num_cluster,))

        result = cur.fetchone()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    #print(result)
    return _typecast_array(result[0])


if __name__ == '__main__':
    connect()
