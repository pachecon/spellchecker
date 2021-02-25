import mysql.connector
import pandas as pd
import sys

from mysql.connector import errorcode

class CreateNgramsDB:
    """
    This class is used to create the database called "ngrams_db".
    The table of the database is called "ngrams2", where contains all the bigrams which are obtained during training and saved as csv file
    
     Attributes
    ----------
    mydb: MySQL.connector
        get information about the host, user and password
    
    Methods
    ---------
    __create_db__()
        Create the database ngrams_db
    __create_tables__()
        Create the table ngram2 into the database ngrams_db where all bigrams are stored
    __load_csv_in_db__
        Insert the csv file into the ngram2 table of the ngrams_db database

    """
    def __init__(self, host, user, passwd, bigramsfilename='../../output/train/baseline_ngrams/bigrams_order.csv'):
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
        filename:str
            The path and name of the file where the bigrams are saved.
            Default '../../output/train/baseline_ngrams/bigrams_order.csv'
        """
        self.mydb = mysql.connector.connect(
                host=host,#"localhost",
                user=user,#"arlette",
                passwd= passwd #"arlette"#"Matthias16"
                )
        self.filename = bigramsfilename

    def __create_db__(self):
        """
        Create the database ngrams_db 
        """
        cursor = self.mydb.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS ngrams_db DEFAULT CHARACTER SET 'utf8'")
        print('Create database ngrams_db')
        cursor.close()

    def __create_tables__(self):
        """
        Create the table ngram2 into the dabase ngrams_db
        We specify the bigrams_key, the index of each bigrams_key and the number of 
        times the bigrams were found during training (as counts_number)
        """
        #ENGINE = INNODB
        self.mydb.connect(database="ngrams_db")
        cursor = self.mydb.cursor()
        cursor.execute("USE ngrams_db" )
        #
        query = ('CREATE TABLE IF NOT EXISTS ngrams2 (id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,' 
                                                    'bigrams_key VARCHAR(255) NOT NULL,' 
                                                    'counts_number INT(11) NOT NULL,'
                                                    'INDEX (bigrams_key))')
        cursor.execute(query)
        cursor.close()

    def __load_csv_in_db__(self):
        """
        Load the csv file were the bigrams are alphabetically stored.
        Insert data into the ngrams2 table 
        """
        cursor = self.mydb.cursor()
        df = pd.read_csv(self.filename)
        for i, b_key in enumerate(df.key): #[1:20]
            print("\rFold {}/{}.".format(i, len(df.key[1:])), end='\r')
            sys.stdout.flush()
            cursor.execute('INSERT INTO ngrams2 (bigrams_key, \
                counts_number )' \
                'VALUES ("%s", "%s");'% 
                (b_key,int(df.value.iloc[i])))
        self.mydb.commit()
        del df
        cursor.close()
    
    def create_all(self):
        """
        The method to be called in order to create the database, table and insert the data from
        the csv file into the table of the database.
        
        """
        print('create database if not exists')
        self.__create_db__()
        print('Connecting with Ngrams database')
        self.__create_tables__()
        print('Saving data from csv file into the database ngrams')
        self.__load_csv_in_db__()
        #close the connection to the database.
        print('Close connection')
        self.mydb.commit()
        print ("Done")



if __name__ == "__main__":
    db = CreateNgramsDB("localhost","arlette","Matthias16") #arlette Matthias16
    db.create_all()
