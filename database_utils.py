import os.path
import sqlite3


def create_database(database_path):
    if not os.path.exists(database_path):
        connection = sqlite3.connect(database_path)

        connection.close()


def create_table(database_path, table_name):
    connection = sqlite3.connect(database_path)

    cursor = connection.cursor()

    cursor.execute(f'''CREATE TABLE {table_name} (
                        frame_index INT NOT NULL,
                        team1_possession VARCHAR(5) NOT NULL,
                        team2_possession VARCHAR(5) NOT NULL
                    )''')

    connection.commit()

    cursor.close()
    connection.close()


def table_exists(database_path, table_name):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    return result is not None


def add_frame_to_table(table_name, frame_index, team1_possession, team2_possession, connection):
    cursor = connection.cursor()

    exe_string = f"INSERT INTO {table_name} (frame_index, team1_possession, team2_possession) VALUES ({frame_index}, " \
                 f"'{team1_possession}', '{team2_possession}')"
    cursor.execute(exe_string)

    connection.commit()

    cursor.close()


def get_frame(table_name, frame_index, connection):
    cursor = connection.cursor()

    cursor.execute(f"SELECT* FROM {table_name} WHERE frame_index=?", (frame_index,))

    result = cursor.fetchone()

    cursor.close()

    if result is not None:
        return result[1], result[2]
    else:
        return None, None
