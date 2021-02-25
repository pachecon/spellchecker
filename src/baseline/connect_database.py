import mysql.connector
import pandas as pd
import sys

from mysql.connector import Error

class NgramsDB:
    """
    This class is used to connect into the database "ngrams_db" for the Noisy Channel Model.

    Attributes
    ----------
    mydb: MySQL.connector
        get information about the host, user and password
    cursor: MySQLConnection.cursor()
        connects to the cursor of MySQL

    Methods
    -------
    __connect_db__():
        connect to the database.
    close_db():
        close connection to the database
    get_data_from_db():
        return records obtained from the database by searching the input query into the table 
        throw exception if there is a problem by searching the query into the table of the database.   
    """
    def __init__(self, host, user, passwd):
        """
        Connect to the database 

        Parameters
        ----------
        host:str
            Example: "localhost"
        user:str
            User name
        passwd:str
            Password of the user
        """
        self.mydb = mysql.connector.connect(
                host=host,      #"localhost",
                user=user,      #"arlette",
                passwd= passwd  #"arlette"#"Matthias16"
                )
        self.__connect_db__()
        self.cursor = self.mydb.cursor()
      
    def __connect_db__(self):
        """
        Connect to the database named "ngrams_db"
        """
        self.mydb.connect(database="ngrams_db")

    def close_db(self):
        """
        Close the connection to the database named "ngrams_db"
        """
        if (self.mydb.is_connected()):
            self.cursor.close()
            self.mydb.close()
    
    def get_data_from_db(self, query):
        """
        Search based on the input query into the table of the database.
        Parameter
        ---------
        query:str
            the query to be searched into the table ngrams2 of the ngrams_db database
        Return
        -------
        list
            List of tuples based on the query result (if not empty); otherwise return 0.0.  
        Exception
        ----------
            Print the error
        """
        try:
            if self.mydb.is_connected():
                self.cursor.execute(query)
                records = self.cursor.fetchall()
                if records:
                    return records
                else:
                    return 0.0
        except Error as e:
            print("Error reading data from MySQL table", e)